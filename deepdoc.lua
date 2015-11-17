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
	self.context_size = config.context_size
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

-- Function to trigger training
function DeepDoc:train()
	print('Training...')
	local start=sys.clock()
	for epoch = 1,self.max_epochs do
		local epoch_start = sys.clock()
		local indices = torch.randperm(#self.index2doc)
		xlua.progress(1, self.index2doc)
		self.tensors = {}
		for i = 1, #self.index2doc do
			
			
			if i % 10 == 0 then
				xlua.progress(i, self.index2doc)		
			end
		end
		xlua.progress(self.index2doc, self.index2doc)
		print(string.format("Epoch %d done in %.2f minutes.\n",epoch,((sys.clock()-epoch_start)/60)))
	end
	print(string.format("Training done in %.2f minutes.",((sys.clock() - start) / 60)))
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
	self.encoder = LSTMEncoder(encode_config)

	-- Define the decoder
	local decode_config = {
		in_dim = self.wdim,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word,
		doc_dim = self.ddim
	}
	self.decoder_left = {}
	self.decoder_right = {}
	for i = 1, self.context_size do 
		table.insert(self.decoder_left, LSTMDecoder(decode_config))
		table.insert(self.decoder_right, LSTMDecoder(decode_config))
	end	

	--[[
	-- encoding
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

	-- Define the decoder
	local decode_config = {
		in_dim = self.wdim,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word,
		doc_dim = self.ddim
	}

	-- decoding
	self.decodeInputModel=nn.ParallelTable()
	self.clone_word_vecs=self.word_vecs:clone("weight","bias","gradWeight","gradBias")
	self.decodeInputModel:add(self.clone_word_vecs)
	self.decodeInputModel:add(self.doc_vecs)
	for i=1, self.num_layers do
		self.decodeInputModel:add(nn.Identity())
	end

	inputs = self.decodeInputModel:forward({torch.Tensor{4, 5}, torch.Tensor{1}, unpack(enc_final_state)})
	outputs = {torch.Tensor{5},torch.Tensor{6}}	
	self.decoder = LSTMDecoder(decode_config)
	dec_out=self.decoder:forward(inputs, outputs)
	grad=self.decoder:backward(inputs, outputs)
	self.decodeInputModel:backward({torch.Tensor{4, 5}, torch.Tensor{1}, unpack(enc_final_state)}, grad)
	]]--
end