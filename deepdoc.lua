--[[

Class for DeepDoc. 

--]]

local DeepDoc = torch.class("DeepDoc")
local utils = require 'utils'

-- Lua Constructor
function DeepDoc:__init(config)
	-- data
	self.data = config.data
	self.pre_train = config.pre_train
	self.pre_train_dir = config.pre_train_dir
	self.to_lower = config.to_lower
	-- model params (general)
	self.wdim = config.wdim
	self.ddim = config.ddim
	self.min_freq = config.min_freq
	self.model = config.model
	self.num_layers = config.num_layers
	self.mem_dim = config.mem_dim
	-- optimization
	self.learning_rate = config.learning_rate
	self.grad_clip = config.grad_clip
	self.batch_size = config.batch_size
	self.max_epochs = config.max_epochs
	self.reg = config.reg
	self.decay = config.decay
	self.dropout = config.dropout
	-- GPU/CPU
	self.gpu = config.gpu

    -- Build vocabulary
	utils.buildVocab(self, false)

	-- Load train set into memory
	utils.loadTensorsToMemory(self)

	-- build the net
    self:build_model()
end

-- Function to build the DeepDoc model
function DeepDoc:build_model()
	-- Define the lookups
	self.word_vecs = nn.LookupTable(#self.index2word, self.wdim)
	self.doc_vecs = nn.LookupTable(#self.index2doc, self.ddim)

	-- Define the encoder
	local encode_config = {
		in_dim = self.wdim,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word
	}

	inputs = self.word_vecs:forward(torch.Tensor{1, 2})
	outputs = {torch.Tensor{2},torch.Tensor{3}}	
	self.encoder = LSTMEncoder(encode_config)	
	enc_out = self.encoder:forward(inputs, outputs)
	enc_final_state = {}
	for i = 2, #enc_out, 2 do
		table.insert(enc_final_state, enc_out[i])
	end
	e_grad=self.encoder:backward(inputs, outputs)
	self.word_vecs:backward(torch.Tensor{1, 2},e_grad)
end

-- Function to build the output model
function DeepDoc:output_model()
	local input_dim = self.num_layers * self.mem_dim
	local inputs,vec
	if self.model == 'gru' then
		local rep = nn.Identity()()
		if self.num_layers == 1 then
			vec = {rep}
		else
			vec = nn.JoinTable(1)(rep)
		end
		inputs = {rep}
	elseif self.structure == 'bi-gru' then
		local frep, brep = nn.Identity()(), nn.Identity()()
		input_dim = input_dim * 2
		if self.num_layers == 1 then
			vec = nn.JoinTable(1){frep, brep}
		else
			vec = nn.JoinTable(1){nn.JoinTable(1)(frep),nn.JoinTable(1)(brep)}
		end
		inputs = {frep, brep}
	end
	local logprobs
	if self.dropout == 1 then
		logprobs = nn.LogSoftMax()(nn.Linear(input_dim,#self.index2word)(nn.Dropout()(vec)))
	else
		logprobs = nn.LogSoftMax()(nn.Linear(input_dim,#self.index2word)(vec))
	end
	return nn.gModule(inputs,{logprobs})
end