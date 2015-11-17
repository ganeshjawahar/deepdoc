--[[

Encoding document through LSTM module
-------------------------------------

Most of this code is adapted from Stanford's TreeLSTM [1] and Char-RNN [2].
1. https://github.com/stanfordnlp/treelstm
2. https://github.com/karpathy/char-rnn

--]]

local LSTMEncoder, parent = torch.class('LSTMEncoder', 'nn.Module')
local utils = require 'utils'

function LSTMEncoder:__init(config)
	parent.__init(self)
	
	self.in_dim = config.in_dim
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
function LSTMEncoder:new_cell()
	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- x
	for L = 1, self.num_layers do
		table.insert(inputs, nn.Identity()()) -- prev_c[L]
		table.insert(inputs, nn.Identity()()) -- prev_h[L]
	end
	
	local outputs = {}
	for L = 1, self.num_layers do
		-- c,h from previous timesteps
		local prev_h = inputs[L * 2 + 1]
		local prev_c = inputs[L * 2]

		local new_gate = function()
			local in_module = (L == 1)
				and nn.Linear(self.in_dim, self.mem_dim)(nn.View(1, self.in_dim)(inputs[1]))
				or  nn.Linear(self.mem_dim, self.mem_dim)(outputs[(L - 1) * 2])
			return nn.CAddTable(){
				in_module,
				nn.Linear(self.mem_dim, self.mem_dim)(prev_h)
			}
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
function LSTMEncoder:forward(word_input, word_output, reverse)
	local size = word_input:size(1)
	self.predictions = {}
	self.rnn_state = {[0] = self.initial_forward_values}
	local loss = 0
	for t = 1, size do
		local input = reverse and word_input[size - t + 1] or word_input[t]
		local label = reverse and word_output[size - t + 1] or word_output[t]
		self.depth = self.depth + 1
		local cell = self.cells[self.depth]
		if cell == nil then
			cell = self:new_cell()
			self.cells[self.depth] = cell
			self.criterions[self.depth] = nn.ClassNLLCriterion()			
		end
		cell:training()
		local lst = cell:forward({input, unpack(self.rnn_state[t - 1])})
		self.rnn_state[t] = {}
		for i=1, 2 * self.num_layers do table.insert(self.rnn_state[t], lst[i]) end
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
function LSTMEncoder:backward(word_input, word_output, reverse)
	local size = word_input:size(1)
	if self.depth == 0 then
		error("No cells to backpropagate through")
	end

	local input_grads = torch.Tensor(word_input:size())
	local drnn_state = {[size] = self.initial_backward_values}
	for t = size, 1, -1 do		
		local input = reverse and word_input[size - t + 1] or word_input[t]
		local label = reverse and word_output[size - t + 1] or word_output[t]
		local doutput_t = self.criterions[self.depth]:backward(self.predictions[t], label)
		local dlst = self.cells[self.depth]:backward({input, unpack(self.rnn_state[t - 1])}, utils.combine(drnn_state[t], doutput_t))
		drnn_state[t - 1] = {}
		for k,v in pairs(dlst) do
			if k > 1 then
				drnn_state[t-1][k-1] = v
			end
		end
		if reverse then
			input_grads[size-t+1] = dlst[1]
		else
			input_grads[t] = dlst[1]
		end
		self.depth = self.depth - 1		
	end
	self.initial_forward_values = self.rnn_state[size] -- transfer final state to initial state (BPTT)
	self:forget() -- important to clear out state
	return input_grads
end

function LSTMEncoder:share(lstm, ...)
	if self.in_dim ~= lstm.in_dim then error("LSTM input dimension mismatch") end
	if self.mem_dim ~= lstm.mem_dim then error("LSTM memory dimension mismatch") end
	if self.num_layers ~= lstm.num_layers then error("LSTM layer count mismatch") end
	if self.dropout ~= lstm.dropout then error("LSTM dropout mismatch") end
	if self.gpu ~= lstm.gpu then error("LSTM gpu state mismatch") end
	if self.vocab_size ~= lstm.vocab_size then error("LSTM vocab size mismatch") end	
	utils.shareParams(self.master_cell, lstm.master_cell,...)
end

function LSTMEncoder:zeroGradParameters()
	self.master_cell:zeroGradParameters()
end

function LSTMEncoder:parameters()
	return self.master_cell:parameters()
end

function LSTMEncoder:forget()
	self.depth = 0
	for i = 1, #self.initial_backward_values do
		self.initial_backward_values[i]:zero()
	end
end