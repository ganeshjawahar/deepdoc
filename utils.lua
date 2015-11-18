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

-- Function to build vocabulary from the corpus
function utils.buildVocab(config)
	print('Building vocabulary...')
	local start = sys.clock()
	config.vocab = {} -- word frequency map
	config.index2word = {}
	config.word2index = {}
	config.index2doc = {}
	config.doc2index = {}

	config.word_count = 0
	local fptr = io.open(config.data, 'r')
	local i = 0
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local ppid, count = unpack(utils.splitByChar(line, ' '))
		config.index2doc[#config.index2doc + 1] = ppid
		config.doc2index[ppid] = #config.index2doc
		for i = 1, count do
			local sentence = fptr:read()
			for _, word in ipairs(utils.padTokens(utils.splitByChar(sentence, ' '))) do
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
		end
		i = i + 1
		if i == 25 then
			break
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
end

-- Function to load sentence tensors to memory
function utils.loadTensorsToMemory(config)
	print('Loading sentence tensors...')
	local start = sys.clock()
	local fptr = io.open(config.data, 'r')
	local lc = 0
	config.sentence_tensors = {}
	config.training_tuples_count = 0
	config.training_docs_count = 0
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local ppid, count = unpack(utils.splitByChar(line, ' '))
		local ppSeq = config.doc2index[ppid]
		config.sentence_tensors[ppSeq] = {}
		for i = 1, count do
			local sentence = fptr:read()
			local tokens = utils.padTokens(utils.splitByChar(sentence, ' '))
			local tensor = torch.Tensor(#tokens)
			for j, word in ipairs(tokens) do
				if config.word2index[word] == nil then
					tensor[j] = config.word2index['<UK>']
				else
					tensor[j] = config.word2index[word]
				end
			end
			local sent_tensor = torch.Tensor{ppSeq}
			if config.gpu == 1 then
				sent_tensor = sent_tensor:cuda()
			end
			table.insert(config.sentence_tensors[ppSeq], {utils.createInputTarget(tensor, config), sent_tensor})
		end
		if #config.sentence_tensors[ppSeq] > (1 + 2 * config.context_size) then
			config.training_tuples_count = config.training_tuples_count + (#config.sentence_tensors[ppSeq] - (2 * config.context_size))
			config.training_docs_count = config.training_docs_count + 1
		end
		lc = lc + 1
		if lc == 25 then
			break
		end
	end
	io.close(fptr)
	print('No. of training tuples = '..config.training_tuples_count)
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
	print(string.format('Done in %.2f minutes', ((sys.clock() - start) / 60)))
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

return utils