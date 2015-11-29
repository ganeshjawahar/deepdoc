require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cunnx'
require 'optim'
require 'xlua'
include('LSTMEncoder.lua')
include('LSTMDecoder.lua')
local utils = require 'utils'

local model = torch.load('model.t7')
model.grad_clip = 10
model.doc_vecs.weight:uniform(-0.05, 0.05)
model.learning_rate = 0.05
model.epoch = 1
model.batch_size = 10

-- Freeze word vecs. learning
model.word_vecs.accGradParameters = function() end

-- Generate dataset
local dataset = {}
local srcFile = io.open('data/abstracts.txt','r')
local doc_idx = {}
while true do
	local line = srcFile:read()
	if line == nil then
		break
	end
	local ppid, count = unpack(utils.splitByChar(line, ' '))
	local sentence_tensors = {}
	for i = 1, tonumber(count) do
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
		local sent_tensor = torch.Tensor{#doc_idx + 1}
		if model.gpu == 1 then
			sent_tensor = sent_tensor:cuda()
		end
		table.insert(sentence_tensors, {utils.createInputTarget(tensor, model), sent_tensor})
	end
	if #sentence_tensors > (1 + 2 * model.context_size) then
		-- Generate tuples	
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
		table.insert(doc_idx, ppid)
	end
	if #doc_idx == 25 then
		break
	end
end
io.close(srcFile)

optim_state = {learningRate = model.learning_rate}
params, grad_params = model.model:getParameters()
feval = function(x)
	-- Get new params
	params:copy(x)

	-- Reset gradients
	grad_params:zero()

	-- loss is average of all criterions
	local loss = 0
	local count = 0
	print(model.start_index..'\t'..model.end_index)
	for i = model.start_index, model.end_index do
		local tuples = dataset[i]
		local sent_id = tuples[1][2]
		-- Do encoding
		local enc_input = tuples[1][1][1]
		local enc_label = tuples[1][1][2]
		local word_inputs = model.word_vecs:forward(enc_input)
		local enc_out, loss0 = model.encoder:forward(word_inputs, enc_label)
		loss = loss + loss0
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
			loss = loss + loss0		
			local misc_grad = model.decoders[i - 1]:backward(dec_word_inputs, dec_label)
			model.decodeInputModel:backward({dec_input, sent_id, unpack(enc_final_state)}, misc_grad)
		end
		count = count + 1
	end
	
	loss = loss / count
	grad_params:div(count)

	-- If the gradients explode, scale down the gradients
	if grad_params:norm() >= model.grad_clip then
		grad_params:mul(model.grad_clip / grad_params:norm())
	end

	return loss, grad_params
end

-- Do testing
for epoch = 1, model.epoch do
	local epoch_start = sys.clock()
	local epoch_iteration, epoch_loss = 0, 0
	xlua.progress(1, #doc_idx)
	for i = 1, #doc_idx, model.batch_size do
		model.start_index = i
		model.end_index = math.min(i + model.batch_size - 1, #doc_idx)		
		local _, loss = optim.adam(feval, params, optim_state)
		epoch_loss = epoch_loss + loss[1]
		epoch_iteration = epoch_iteration + 1
	end
	xlua.progress(#doc_idx, #doc_idx)	
	print(string.format("Epoch %d done in %.2f minutes. Loss = %f\n",epoch,((sys.clock() - epoch_start) / 60), (epoch_loss / epoch_iteration)))
end

-- Save the predicted document representations
local docFile = io.open('doc_test.txt','w')
for i = 1, #doc_idx do
	docFile:write(doc_idx[i]..'\t'..utils.convert_to_string(model.doc_vecs.weight[i])..'\n')
end
io.close(docFile)