--[[
Utility function used by DeepDoc class.
--]]

local utils = {}

-- Function to trim the string
function utils.trim(s)
  return (s:gsub("^%s*(.-)%s*$",  "%1"))
end

-- Function to pad tokens.
function utils.padTokens(tokens)
	local res = {}

	-- Append begin token
	table.insert(res, '<bos>')

	for _, word in ipairs(tokens) do
		table.insert(res, word)
	end

	-- Append end tokens
	table.insert(res, '<eos>')

	return res
end

-- Function to split a string by given char.
function utils.splitByChar(str, inSplitPattern)
	str = utils.trim(str)
	outResults = {}
	local theStart  =  1
	local theSplitStart, theSplitEnd = string.find(str, inSplitPattern, theStart)
	while theSplitStart do
		table.insert(outResults, string.sub(str, theStart, theSplitStart-1))
		theStart = theSplitEnd + 1
		theSplitStart, theSplitEnd = string.find(str, inSplitPattern, theStart)
	end
	table.insert(outResults, string.sub(str, theStart))
	return outResults
end

-- Function to build document vocab
function utils.buildDoc(config)
	local fptr = io.open(config.data, 'r')
	config.index2doc = {}
	config.doc2index = {}
	config.training_tuples_count = 0
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local ppid, count = unpack(utils.splitByChar(line, '\t'))		
		count = tonumber(count)
		local is_valid_doc = true
		for i = 1, count do
			local sentence = fptr:read()
			local words = utils.padTokens(utils.splitByChar(sentence, ' '))
			if #words > config.max_sent_size then
				is_valid_doc = false
			end
		end		
		if count > (1 + 2 * config.context_size) and is_valid_doc == true then
			config.index2doc[#config.index2doc + 1] = ppid
			config.doc2index[ppid] = #config.index2doc			
			config.training_tuples_count = config.training_tuples_count + (count - (2 * config.context_size))
		end	
		if #config.index2doc == 100000 then
			break
		end
	end
	io.close(fptr)	
end

-- Function to build vocabulary from the corpus
function utils.buildVocab(config)
	print('Building vocabulary...')
	local start = sys.clock()
	config.vocab = {} -- word frequency map
	config.index2word = {}
	config.word2index = {}
	utils.buildDoc(config)

	config.word_count = 0
	local fptr = io.open(config.data, 'r')
	config.max_sent_size = 0
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local ppid, count = unpack(utils.splitByChar(line, '\t'))
		local local_max = 0
		for i = 1, count do
			local sentence = fptr:read()
			if config.doc2index[ppid] ~= nil then
				local tokens = utils.padTokens(utils.splitByChar(sentence, ' '))
				for _, word in ipairs(tokens) do
					if config.to_lower == 1 then
						word = word:lower()
					end

					-- Fill word vocab.
					if config.vocab[word] == nil then
						config.vocab[word] = 1
					else
						config.vocab[word] = config.vocab[word] + 1
					end
					config.word_count = config.word_count+1
				end
				if #tokens > config.max_sent_size then
					config.max_sent_size = #tokens
				end
			end			
		end
	end
	io.close(fptr)	

	-- Discard the words that doesn't meet minimum frequency and create indices.
	for word, count in pairs(config.vocab) do
		if count < config.min_freq then
			config.vocab[word] = nil
		else
			config.index2word[#config.index2word + 1] = word
			config.word2index[word] = #config.index2word
		end
	end

	-- Add unknown word
	config.vocab['<UK>'] = 1
	config.index2word[#config.index2word + 1] = '<UK>'
	config.word2index['<UK>'] = #config.index2word

	print(string.format("%d words, %d documents processed in %.2f minutes.", config.word_count, #config.index2doc, ((sys.clock() - start) / 60)))
	print(string.format("Vocab size after eliminating words occuring less than %d times: %d", config.min_freq, #config.index2word))
	print('No. of training tuples = '..config.training_tuples_count)
	print('Global Max. Sent Size = '..config.max_sent_size..' (including pads)')
end

-- Function to load sentence tensors to memory
function utils.loadTensorsToMemory(config)
	print('Loading sentence tensors...')
	local start = sys.clock()
	local fptr = io.open(config.data, 'r')
	config.sentence_tensors = {}
	xlua.progress(1, #config.index2doc)
	local train_count = 0
	local count = 0
	local lc = 0
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local ppid, count = unpack(utils.splitByChar(line, '\t'))
		local tensors = {}
		for i = 1, count do
			local sentence = fptr:read()
			if config.doc2index[ppid] ~= nil then
				local tokens = utils.padTokens(utils.splitByChar(sentence, ' '))
				local tensor = torch.Tensor(#tokens)
				for j, word in ipairs(tokens) do
					if config.word2index[word] == nil then
						tensor[j] = config.word2index['<UK>']
					else
						tensor[j] = config.word2index[word]
					end
				end
				local sent_tensor = torch.Tensor{config.doc2index[ppid]}
				if config.gpu == 1 then
					sent_tensor = sent_tensor:cuda()
				end
				table.insert(tensors, {utils.createInputTarget(tensor, config), sent_tensor})
			end
		end		
		if config.doc2index[ppid] ~= nil then
			train_count = train_count + 1
			config.sentence_tensors[config.doc2index[ppid]] = tensors
			lc = lc + 1
		else
			tensors = nil
		end
		if train_count % 100 == 0 then
			xlua.progress(train_count, #config.index2doc)
			collectgarbage()
		end
		if lc == 100000 then
			break
		end
	end
	xlua.progress(#config.index2doc, #config.index2doc)
	io.close(fptr)
	print(string.format('Done in %.2f minutes', ((sys.clock() - start) / 60)))
end

-- Function to load dataset into CPU RAM.
function utils.loadStringsToMemory(config)
	print('Loading sentence strings ...')
	local start = sys.clock()
	local fptr = io.open(config.data, 'r')
	config.sentence_strings = {}
	xlua.progress(1, #config.index2doc)
	local train_count = 0
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local ppid, count = unpack(utils.splitByChar(line, '\t'))
		local strings = {}
		local ppSeq = config.doc2index[ppid]
		for i = 1, count do
			local sentence = fptr:read()
			table.insert(strings, sentence)
		end
		if config.doc2index[ppid] ~= nil then
			train_count = train_count + 1
			config.sentence_strings[config.doc2index[ppid]] = strings
		else
			strings = nil
		end
		if train_count % 100 == 0 then
			xlua.progress(train_count, #config.index2doc)
			collectgarbage()
		end
	end
	xlua.progress(#config.index2doc, #config.index2doc)
	io.close(fptr)
	print(string.format('Done in %.2f minutes', ((sys.clock() - start) / 60)))
end

-- share module parameters
function utils.shareParams(cell, src)
	if torch.type(cell) == 'nn.gModule' then
		for i = 1, #cell.forwardnodes do
		local node = cell.forwardnodes[i]
		if node.data.module then
			node.data.module:share(src.forwardnodes[i].data.module, 
				'weight', 'bias', 'gradWeight', 'gradBias')
		end
	end
	elseif torch.isTypeOf(cell, 'nn.Module') then  	
		cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
	else
		error('parameters cannot be shared for this input')
	end
end

-- select sub-range from table
function utils.tableSelect(input, start, last)
	if 1 <= start and last <= #input then
		local resTable = {}
		for i = start, last do
			table.insert(resTable, input[i])
		end
		return resTable
	else
		error('invalid indices for selecting the table')
	end
end

-- Function to create input and target tensor for one sentence
function utils.createInputTarget(tensor, config)
	local data_tensor = torch.Tensor(tensor:size(1) - 1)
	local target_tensor = torch.IntTensor(tensor:size(1) - 1, 1)
	for i = 1, tensor:size(1) - 1 do
		data_tensor[i] = tensor[i]
		target_tensor[i] = tensor[i + 1]
	end
	if config.gpu == 1 then
		data_tensor = data_tensor:cuda()
		target_tensor = target_tensor:cuda()
	end
	return {data_tensor, target_tensor}
end

-- Function to combine table and userdata
function utils.combine(tab, ud)
	local resTable = {}
	for _,data in ipairs(tab) do
		table.insert(resTable, data)
	end
	table.insert(resTable, ud)
	return resTable
end

-- Function to build frequency-based tree for Hierarchical Softmax
function utils.create_frequency_tree(freq_map)
	binSize=100
	print('Creating frequency tree with '..binSize..' as bin size...')
	local start = sys.clock()
	local ft = torch.IntTensor(freq_map)
	local vals, indices = ft:sort()
	local tree = {}
	local id = indices:size(1)
	function recursiveTree(indices)
		if indices:size(1) < binSize then
			id = id + 1
			tree[id] = indices
			return
		end
		local parents = {}
		for start = 1, indices:size(1), binSize do
			local stop = math.min(indices:size(1), start + binSize - 1)
			local bin = indices:narrow(1, start, stop - start + 1)
			assert(bin:size(1) <= binSize)
			id = id + 1
			table.insert(parents, id)
			tree[id] = bin
		end
		recursiveTree(indices.new(parents))
	end
	recursiveTree(indices)	
	return tree, id
end

-- Function to create word map (for Softmaxtree)
function utils.create_word_map(vocab,index2word)
	word_map = {}
	for i=1, #index2word do
		word_map[i] = vocab[index2word[i]]
	end
	return word_map
end

-- Function to convert tensor to string
function utils.convert_to_string(tensor)
	local res = ''
	for i = 1, tensor:size(1) do
		res = res .. tensor[i] .. '\t'
	end	
	return utils.trim(res)
end

-- Function to create input batches
function utils.createBatches(config)
	print('Creating batches for lazy-loading...')
	local start = sys.clock()
	local fptr = io.open(config.data, 'r')
	config.rootDir = 'batches/'
	local maxTuplesPerFile, proc_tup_count = 50000, 0
	config.seqNo = 1
	local dataset_batches, dataset = {}, {}
	if path.exists(config.rootDir) then lfs.rmdir(config.rootDir) end
	lfs.mkdir(config.rootDir)
	xlua.progress(1, config.training_tuples_count)
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local ppid, count = unpack(utils.splitByChar(line, '\t'))
		local sent_tensors = {}
		for i = 1, count do
			local sentence = fptr:read()
			if config.doc2index[ppid] ~= nil then
				local tokens = utils.padTokens(utils.splitByChar(sentence, ' '))
				local tensor = torch.Tensor(#tokens)
				for j, word in ipairs(tokens) do
					if config.word2index[word] == nil then
						tensor[j] = config.word2index['<UK>']
					else
						tensor[j] = config.word2index[word]
					end
				end
				local sent_tensor = torch.Tensor{config.doc2index[ppid]}
				if config.gpu == 1 then
					sent_tensor = sent_tensor:cuda()
				end
				table.insert(sent_tensors, {utils.createInputTarget(tensor, config), sent_tensor})
			end
		end
		if config.doc2index[ppid] ~= nil then
			for j = (1 + config.context_size), (#sent_tensors - config.context_size) do
				local input = {}
				table.insert(input, sent_tensors[j])
				for k = (j - config.context_size), (j + config.context_size) do
					if k ~= j then
						table.insert(input, sent_tensors[k])
					end
				end
				table.insert(dataset, input)
				if #dataset == config.batch_size then
					table.insert(dataset_batches, dataset)
					dataset = nil
					dataset = {}
				end
				if proc_tup_count % maxTuplesPerFile == 0 then
					torch.save(config.rootDir..'b_'..config.seqNo..'.t7', dataset_batches)
					dataset_batches = nil
					dataset_batches = {}
					collectgarbage()
					config.seqNo = config.seqNo + 1
				end
				proc_tup_count = proc_tup_count + 1
				if proc_tup_count % 300 == 0 then
					xlua.progress(proc_tup_count, config.training_tuples_count)
				end
			end
		else
			sent_tensors = nil
		end
	end
	if #dataset ~= 0 then
		table.insert(dataset_batches, dataset)
		torch.save(config.rootDir..'b_'..config.seqNo..'.t7', dataset_batches)
		dataset = nil
		dataset = {}
		dataset_batches = nil
		dataset_batches = {}
		collectgarbage()
		config.seqNo = config.seqNo + 1
	end
	config.seqNo = config.seqNo - 1
	xlua.progress(config.training_tuples_count, config.training_tuples_count)
	io.close(fptr)
	print(string.format('Done in %.2f minutes', ((sys.clock() - start) / 60)))
end

-- Function to initalize word weights
function utils.initWordWeights(config)
	print('initializing the pre-trained embeddings...')
	local start=sys.clock()
	local ic=0
	for line in io.lines(config.pre_train_file) do
		local content=utils.splitByChar(line,' ')
		local word=content[1]
		if config.word2index[word]~=nil then
			local tensor=torch.Tensor(#content-1)
			for i=2,#content do
				tensor[i-1]=tonumber(content[i])
			end
			config.word_vecs.weight[config.word2index[word]]=tensor
			ic=ic+1
		end
	end
	print(string.format("%d out of %d words initialized.",ic,#config.index2word))
	print(string.format("Done in %.2f seconds.",sys.clock()-start))
end

return utils