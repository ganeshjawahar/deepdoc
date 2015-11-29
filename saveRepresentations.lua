require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cunnx'
include('LSTMEncoder.lua')
include('LSTMDecoder.lua')
local utils = require 'utils'

local model = torch.load('model.t7')

-- write document embeddings
local docFile = io.open('doc_1L.txt','w')
for i = 1, #model.index2doc do
	docFile:write(model.index2doc[i]..'\t'..utils.convert_to_string(model.doc_vecs.weight[i])..'\n')
end
io.close(docFile)

-- write word embeddings
local wordFile = io.open('word_1L.txt','w')
for i = 1, #model.index2word do
	wordFile:write(model.index2word[i]..'\t'..utils.convert_to_string(model.word_vecs.weight[i])..'\n')
end
io.close(wordFile)