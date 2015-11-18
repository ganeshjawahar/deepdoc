--[[

DeepDoc: Learning Document Representations using Skip-Thought Vectors

]]--

require 'torch'
require 'io'
require 'nn'
require 'nngraph'
require 'sys'
require 'optim'
require 'os'
require 'xlua'
require 'lfs'
include('deepdoc.lua')
include('LSTMEncoder.lua')
include('LSTMDecoder.lua')

cmd = torch.CmdLine()
cmd:text()
cmd:text('DeepDoc: Learning Document Representations using Skip-Thought Vectors')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data','data/abstracts.txt','Document Directory.')
cmd:option('-pre_train',1,'initialize word embeddings with pre-trained vectors?')
cmd:option('-pre_train_dir','data/','Directory for accesssing the pre-trained word embeddings')
cmd:option('-to_lower',1,'change the case of word to lower case')
cmd:option('-max_test_size',10,'maximum no. of testing documents to expect')
-- model params (general)
cmd:option('-wdim',100,'dimensionality of word embeddings')
cmd:option('-ddim',200,'dimensionality of document embeddings')
cmd:option('-min_freq',5,'words that occur less than <int> times will not be taken for training')
cmd:option('-model','lstm','LSTM variant to train (lstm, bi-lstm)')
cmd:option('-num_layers',3,'number of layers in LSTM')
cmd:option('-mem_dim',150,'LSTM memory dimensions')
cmd:option('-context_size',1,'Context size')
-- optimization
cmd:option('-learning_rate',0.001,'learning rate')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-batch_size',10,'number of sequences to train on in parallel')
cmd:option('-max_epochs',5,'number of full passes through the training data')
cmd:option('-decay',0.95,'decay rate for adam')
cmd:option('-dropout',0.5,'dropout for regularization, used after each LSTM hidden layer. 0 = no dropout')
cmd:option('-softmaxtree',1,'use SoftmaxTree instead of the inefficient (full) softmax')
-- GPU/CPU
cmd:option('-gpu',1,'1=use gpu; 0=use cpu;')
-- Book-keeping
cmd:option('-print_params',0,'output the parameters in the console. 0=dont print; 1=print;')

-- parse input params
params = cmd:parse(arg)

if params.print_params == 1 then
	-- output the parameters	
	for param, value in pairs(params) do
	    print(param ..' : '.. tostring(value))
	end
end

-- load cuda libraries
if params.gpu == 1 then
	require 'cunn'
	require 'cutorch'
end

model=DeepDoc(params)
model:train()
model:save_model()