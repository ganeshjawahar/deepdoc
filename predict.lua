require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cunnx'
include('LSTMEncoder.lua')
include('LSTMDecoder.lua')
local utils = require 'utils'

model = torch.load('model.t7')
srcFile = io.open('data/abstracts.txt','r')
destFile = io.open('doc_embeddings.txt','w')
local doc_idx = model.training_docs_count + 1

-- Freeze word vecs. learning
model.word_vecs.accGradParameters = function() end

while true do
	local line = srcFile:read()
	if line == nil then
		break
	end
	local ppid, count = unpack(utils.splitByChar(line, ' '))
	local sentence_tensors = {}
	for i = 1, count do
		local sentence = srcFile:read()
		local tokens = utils.padTokens(utils.splitByChar(sentence, ' '))
		local tensor = torch.Tensor(#tokens)
		for j, word in ipairs(tokens) do
			if model.word2index[word] == nil then
				tensor[j] = model.word2index['<UK>']
			else
				tensor[j] = model.word2index[word]
			end
		end
		local sent_tensor = torch.Tensor{doc_idx}
		if model.gpu == 1 then
			sent_tensor = sent_tensor:cuda()
		end
		table.insert(sentence_tensors, {utils.createInputTarget(tensor, model), sent_tensor})
	end
	if #sentence_tensors > (1 + 2 * model.context_size) then
		-- Generate tuples		
		local dataset = {}
		for j = (1 + model.context_size), (#sentence_tensors - model.context_size) do
			local input = {}
			table.insert(input, sentence_tensors[j])
			for k = (j - model.context_size), (j + model.context_size) do
				if k ~= j then
					table.insert(input, sentence_tensors[k])
				end
			end
			table.insert(dataset, input)
		end
		-- Do inference
		for _, tuples in ipairs(dataset) do
			local sent_id = tuples[1][2]
			-- Do encoding
			local enc_input = tuples[1][1][1]
			local enc_label = tuples[1][1][2]
			local word_inputs = model.word_vecs:forward(enc_input)
			local enc_out, loss0 = model.encoder:forward(word_inputs, enc_label)
			local enc_final_state = {}
			for i = 2, #enc_out, 2 do
				table.insert(enc_final_state, enc_out[i])
			end
			local word_grad = model.encoder:backward(word_inputs, enc_label)
			model.word_vecs:backward(enc_input, word_grad)

			-- Do decoding
			for i = 2, (2 * model.context_size + 1) do
				local dec_input = tuples[i][1][1]
				local dec_label = tuples[i][1][2]
				local dec_word_inputs = model.decodeInputModel:forward({dec_input, sent_id, unpack(enc_final_state)})
				local dec_out, loss0 = model.decoders[i - 1]:forward(dec_word_inputs, dec_label)
				local misc_grad = model.decoders[i - 1]:backward(dec_word_inputs, dec_label)
				model.decodeInputModel:backward({dec_input, sent_id, unpack(enc_final_state)}, misc_grad)
			end
		end
		-- Write the document embedding
		print('Writing representations for doc. '..doc_idx..' ('..ppid..')')
		destFile:write(ppid..'\n')
		destFile:write(utils.convert_to_string(model.doc_vecs.weight[doc_idx])..'\n')
		doc_idx = doc_idx + 1
		if doc_idx == 30 then
			break
		end
	end		
end
io.close(srcFile)
io.close(destFile)