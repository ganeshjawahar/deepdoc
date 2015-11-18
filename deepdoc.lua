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
	self.max_test_size = config.max_test_size
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
	self.softmaxtree = config.softmaxtree
	-- GPU/CPU
	self.gpu = config.gpu

    -- Build vocabulary
	utils.buildVocab(self)

	if self.softmaxtree == 1 then
		-- Create frequency based tree
		require 'nnx'
		self.tree, self.root = utils.create_frequency_tree(utils.create_word_map(self.vocab, self.index2word))
		if self.gpu == 1 then
			require 'cunnx'
		end
	end

	-- Load train set into memory
	utils.loadTensorsToMemory(self)

	-- build the net
    self:build_model()
end

-- Function to trigger training
function DeepDoc:train()
	print('Training...')
	local start = sys.clock()
	self:define_feval()
	for epoch = 1, self.max_epochs do
		local epoch_start = sys.clock()
		local indices = torch.randperm(#self.index2doc)
		local epoch_loss, epoch_iteration = 0, 0
		xlua.progress(1, self.training_tuples_count)
		self.dataset = {}
		local proc_tup_count = 0
		for i = 1, #self.index2doc do
			local idx = indices[i]
			local sent_tensors = self.sentence_tensors[idx]
			if #sent_tensors > (1 + 2 * self.context_size) then
				for j = (1 + self.context_size), (#sent_tensors - self.context_size) do
					local input = {}
					table.insert(input, sent_tensors[j])
					for k = (j - self.context_size), (j + self.context_size) do
						if k ~= j then
							table.insert(input, sent_tensors[k])
						end
					end
					table.insert(self.dataset, input)
					if #self.dataset == self.batch_size then
						local _, loss = optim.adam(self.feval, self.params, self.optim_state)
						epoch_loss = epoch_loss + loss[1]
						epoch_iteration = epoch_iteration + 1
						self.dataset = {}
					end
					proc_tup_count = proc_tup_count + 1
					if proc_tup_count % 5 == 0 then
						xlua.progress(proc_tup_count, self.training_tuples_count)
					end
				end
			end
			if i % 10 == 0 then
				collectgarbage()	
			end
		end		
		if #self.dataset ~= 0 then
			local _, loss = optim.adam(self.feval, self.params, self.optim_state)
			epoch_loss = epoch_loss + loss[1]
			epoch_iteration = epoch_iteration + 1
		end
		xlua.progress(self.training_tuples_count, self.training_tuples_count)
		print(string.format("Epoch %d done in %.2f minutes. Loss = %f\n",epoch,((sys.clock() - epoch_start) / 60), (epoch_loss / epoch_iteration)))
	end
	print(string.format("Training done in %.2f minutes.",((sys.clock() - start) / 60)))
end

-- Function to define feval
function DeepDoc:define_feval()
	self.optim_state = {learningRate = self.learning_rate, alpha = self.decay}
	self.params, self.grad_params = self.model:getParameters()
	self.feval = function(x)
		-- Get new params
		self.params:copy(x)

		-- Reset gradients
		self.grad_params:zero()

		-- loss is average of all criterions
		local loss=0
		for _, tuples in ipairs(self.dataset) do
			local sent_id = tuples[1][2]
			-- Do encoding
			local enc_input = tuples[1][1][1]
			local enc_label = tuples[1][1][2]
			local word_inputs = self.word_vecs:forward(enc_input)
			local enc_out, loss0 = self.encoder:forward(word_inputs, enc_label)
			loss = loss + loss0
			local enc_final_state = {}
			for i = 2, #enc_out, 2 do
				table.insert(enc_final_state, enc_out[i])
			end
			local word_grad = self.encoder:backward(word_inputs, enc_label)
			self.word_vecs:backward(enc_input, word_grad)

			-- Do decoding
			for i = 2, (2 * self.context_size + 1) do
				local dec_input = tuples[i][1][1]
				local dec_label = tuples[i][1][2]
				local dec_word_inputs = self.decodeInputModel:forward({dec_input, sent_id, unpack(enc_final_state)})
				local dec_out, loss0 = self.decoders[i - 1]:forward(dec_word_inputs, dec_label)
				loss = loss + loss0		
				local misc_grad = self.decoders[i - 1]:backward(dec_word_inputs, dec_label)
				self.decodeInputModel:backward({dec_input, sent_id, unpack(enc_final_state)}, misc_grad)
			end
		end
		
		loss = loss / #self.dataset
		self.grad_params:div(#self.dataset)

		-- If the gradients explode, scale down the gradients
		if self.grad_params:norm() >= self.grad_clip then
			self.grad_params:mul(self.grad_clip / self.grad_params:norm())
		end

		return loss, self.grad_params
	end
end


-- Function to build the DeepDoc model
function DeepDoc:build_model()
	-- Define the lookups
	self.word_vecs = nn.LookupTable(#self.index2word, self.wdim)
	self.doc_vecs = nn.LookupTable(#self.index2doc + self.max_test_size, self.ddim)
	if self.gpu == 1 then 
		self.word_vecs = self.word_vecs:cuda()
		self.doc_vecs = self.doc_vecs:cuda()
	end
	self.model = nn.Parallel()

	-- Define the encoder
	local encode_config = {
		in_dim = self.wdim,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word,
		tree = self.tree,
		root = self.root,
		softmaxtree = self.softmaxtree
	}
	self.encoder = LSTMEncoder(encode_config)
	self.model:add(self.encoder)
	if self.gpu == 1 then self.encoder = self.encoder:cuda() end

	-- Define the decoder
	local decode_config = {
		in_dim = self.wdim,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word,
		doc_dim = self.ddim,
		tree = self.tree,
		root = self.root,		
		softmaxtree = self.softmaxtree
	}
	self.decoders = {}
	for i = 1, self.context_size do 
		local decoder_l, decoder_r = LSTMDecoder(decode_config), LSTMDecoder(decode_config)
		table.insert(self.decoders, decoder_l)
		table.insert(self.decoders, decoder_r)
		self.model:add(decoder_l)
		self.model:add(decoder_r)
		if self.gpu == 1 then 
			decoder_l = decoder_l:cuda() 
			decoder_r = decoder_r:cuda()
		end
	end
	self.decodeInputModel = nn.ParallelTable()
	self.clone_word_vecs = self.word_vecs:clone("weight", "bias", "gradWeight", "gradBias")
	self.decodeInputModel:add(self.clone_word_vecs)
	self.decodeInputModel:add(self.doc_vecs)
	for i = 1, self.num_layers do
		self.decodeInputModel:add(nn.Identity())
	end
	if self.gpu == 1 then self.decodeInputModel = self.decodeInputModel:cuda() end

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

-- Save the model
function DeepDoc:save_model()
	print('Saving the model...')
	local start = sys.clock()
	local info = {}
	info.word_vecs = self.word_vecs
	info.doc_vecs = self.doc_vecs
	info.model = self.model
	info.encode_config = self.encode_config
	info.encoder = self.encoder
	info.decode_config = self.decode_config
	info.decoders = self.decoders
	info.decodeInputModel = self.decodeInputModel
	info.vocab = self.vocab
	info.index2word = self.index2word
	info.word2index = self.word2index
	info.index2doc = self.index2doc
	info.doc2index = self.doc2index
	if self.softmaxtree == 1 then
		info.tree = self.tree
		info.root = self.root
	end
	info.gpu = self.gpu
	info.context_size = self.context_size
	info.training_docs_count = self.training_docs_count
	torch.save('model.t7', info)
	print(string.format('Done in %.2f minutes', ((sys.clock() - start) / 60)))
end