--[[

Decoding document through LSTM module
-------------------------------------

Most of this code is adapted from Stanford's TreeLSTM [1] and Char-RNN [2].
1. https://github.com/stanfordnlp/treelstm
2. https://github.com/karpathy/char-rnn

--]]

local LSTMDecoder, parent = torch.class('LSTMDecoder', 'nn.Module')
local utils = require 'utils'

function LSTMDecoder:__init(config)
	parent.__init(self)
	
	self.in_dim = config.in_dim
	self.doc_dim = config.doc_dim
	self.mem_dim = config.mem_dim
	self.num_layers = config.num_layers
	self.dropout=config.dropout
	self.gpu = config.gpu
	self.vocab_size = config.vocab_size

	self.master_cell = self:new_cell()
	self.depth = 0
	self.cells = {} -- table of cells in a roll-out
	self.criterions = {} -- table of criterions
	
	-- initial (t  =  0) states for forward propagation and initial error signals for backpropagation
	self.initial_forward_values, self.initial_backward_values = {}, {}
	for i = 1, self.num_layers do
		table.insert(self.initial_forward_values, torch.zeros(self.mem_dim)) -- c[i]
		table.insert(self.initial_forward_values, torch.zeros(self.mem_dim)) -- h[i]
		table.insert(self.initial_backward_values, torch.zeros(self.mem_dim)) -- c[i]
		table.insert(self.initial_backward_values, torch.zeros(self.mem_dim)) -- h[i]
	end
end

-- Instantiate a new LSTM cell.
-- Each cell shares the same parameters, but the activations of their constituent layers differ.
function LSTMDecoder:new_cell()
	-- there will be 1+2*n+1+n inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- x

	for L = 1, self.num_layers do
		table.insert(inputs, nn.Identity()()) -- prev_c[L]
		table.insert(inputs, nn.Identity()()) -- prev_h[L]
	end

	-- Document embedding
	table.insert(inputs, nn.Identity()())
	
	-- Encoder final hidden state
	for L = 1, self.num_layers do
		table.insert(inputs, nn.Identity()())
	end

	local outputs = {}
	for L = 1, self.num_layers do
		-- c, h from previous timesteps
		local prev_h = inputs[L * 2 + 1]
		local prev_c = inputs[L * 2]

		local new_gate = function()
			local in_module = (L == 1)
				and nn.Linear(self.in_dim, self.mem_dim)(nn.View(1, self.in_dim)(inputs[1]))
				or  nn.Linear(self.mem_dim, self.mem_dim)(outputs[(L - 1) * 2])
			local old_part = nn.CAddTable(){			
				in_module,
				nn.Linear(self.mem_dim, self.mem_dim)(prev_h)
			}
			local new_part = nn.CAddTable(){			
				nn.Linear(self.mem_dim, self.mem_dim)(inputs[self.num_layers * 2 + 2 + L]),
				nn.Linear(self.doc_dim, self.mem_dim)(inputs[self.num_layers * 2 + 2])
			}
			return nn.CAddTable(){old_part, new_part}
		end

		-- decode the gates (input, forget, and output gates)
		local i = nn.Sigmoid()(new_gate())
		local f = nn.Sigmoid()(new_gate())
		local o = nn.Sigmoid()(new_gate())
		-- decode the write inputs
		local update = nn.Tanh()(new_gate())
		-- perform the LSTM update
		local next_c = nn.CAddTable(){
			nn.CMulTable(){f, prev_c},
			nn.CMulTable(){i, update}
		}
		-- gated cells form the output
		local next_h = nn.CMulTable(){o, nn.Tanh()(next_c)}

		table.insert(outputs, next_c)
		table.insert(outputs, next_h)
	end

	-- set up the word prediction
	local top_h = outputs[#outputs]
	if self.dropout > 0 then top_h = nn.Dropout(self.dropout)(top_h) end
	local proj = nn.Linear(self.mem_dim, self.vocab_size)(top_h):annotate{name = 'decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)

	local cell = nn.gModule(inputs, outputs)
	-- share parameters
	if self.master_cell then
		utils.shareParams(cell, self.master_cell)
	end

	if self.gpu == 1 then
		cell = cell:cuda()
	end

	return cell
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional LSTMs).
-- Returns the final hidden state of the LSTM.
function LSTMDecoder:forward(inputs, word_output, reverse)
	local size = inputs[1]:size(1)
	local doc_embed_plus_enc_state = utils.tableSelect(inputs, 2, #inputs)
	self.predictions = {}
	self.rnn_state = {[0] = self.initial_forward_values}
	local loss = 0
	for t = 1, size do
		local input = reverse and inputs[1][size - t + 1] or inputs[1][t]
		local label = reverse and word_output[size - t + 1] or word_output[t]
		self.depth = self.depth + 1
		local cell = self.cells[self.depth]
		if cell == nil then
			cell = self:new_cell()
			self.cells[self.depth] = cell
			self.criterions[self.depth] = nn.ClassNLLCriterion()			
		end
		cell:training()
		local lst = cell:forward(self:get_decoder_input(input, self.rnn_state[t - 1], doc_embed_plus_enc_state))
		self.rnn_state[t] = {}
		for i = 1, 2 * self.num_layers do table.insert(self.rnn_state[t], lst[i]) end
		self.predictions[t] = lst[#lst]
		loss = loss + self.criterions[self.depth]:forward(self.predictions[t], label)
		self.output = lst
	end
	return self.output, loss
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- reverse: if true, read the input from right to left.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function LSTMDecoder:backward(inputs, word_output, reverse)
	if self.depth == 0 then
		error("No cells to backpropagate through")
	end

	local size = inputs[1]:size(1)
	local doc_embed_plus_enc_state = utils.tableSelect(inputs, 2, #inputs)
	local input_word_grads = torch.Tensor(inputs[1]:size())
	local input_doc_grads = torch.zeros(self.doc_dim)
	local drnn_state = {[size] = self.initial_backward_values}	
	for t = size, 1, -1 do		
		local input = reverse and inputs[1][size - t + 1] or inputs[1][t]
		local label = reverse and word_output[size - t + 1] or word_output[t]
		local doutput_t = self.criterions[self.depth]:backward(self.predictions[t], label)
		local dlst = self.cells[self.depth]:backward(self:get_decoder_input(input, self.rnn_state[t - 1], doc_embed_plus_enc_state), utils.combine(drnn_state[t], doutput_t))
		drnn_state[t - 1] = {}
		for k,v in pairs(dlst) do
			if 2 <= k and k <= (1 + 2 * self.num_layers) then
				drnn_state[t - 1][k - 1] = v
			end
		end
		-- update the word embedding input grads
		if reverse then
			input_word_grads[size - t + 1] = dlst[1]
		else
			input_word_grads[t] = dlst[1]
		end
		-- update the document embedding input grads
		input_doc_grads:add(dlst[2 + 2 * self.num_layers])
		self.depth = self.depth - 1		
	end
	self.initial_forward_values = self.rnn_state[size] -- transfer final state to initial state (BPTT)

	local input_grad = {}
	table.insert(input_grad, input_word_grads)
	table.insert(input_grad, input_doc_grads)
	for i = 1, self.num_layers do
		-- Generate some filler grads for encoder states (since it need not be updated) 
		table.insert(input_grad, torch.Tensor(self.mem_dim))
	end
	self:forget() -- important to clear out state
	return input_grad
end

function LSTMDecoder:share(lstm, ...)
	if self.in_dim ~= lstm.in_dim then error("LSTM input dimension mismatch") end
	if self.mem_dim ~= lstm.mem_dim then error("LSTM memory dimension mismatch") end
	if self.num_layers ~= lstm.num_layers then error("LSTM layer count mismatch") end
	if self.dropout ~= lstm.dropout then error("LSTM dropout mismatch") end
	if self.gpu ~= lstm.gpu then error("LSTM gpu state mismatch") end
	if self.vocab_size ~= lstm.vocab_size then error("LSTM vocab size mismatch") end	
	utils.shareParams(self.master_cell, lstm.master_cell,...)
end

function LSTMDecoder:zeroGradParameters()
	self.master_cell:zeroGradParameters()
end

function LSTMDecoder:parameters()
	return self.master_cell:parameters()
end

function LSTMDecoder:get_decoder_input(input, rnn_state, doc_embed_plus_enc_state)	
	local resTable={}
	table.insert(resTable, input)
	for _,state in ipairs(rnn_state) do
		table.insert(resTable, state)
	end
	for _,state in ipairs(doc_embed_plus_enc_state) do
		table.insert(resTable, state)
	end
	return resTable
end

function LSTMDecoder:forget()
	self.depth = 0
	for i = 1, #self.initial_backward_values do
		self.initial_backward_values[i]:zero()
	end
end