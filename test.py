import torch


x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0)
z = torch.tensor(3.0)
w = torch.tensor(5.0)

out = 6 * x + y
out = 3 * w + out

print(out)


out.backward()

print(x.grad)



print('HelloWorld')


subnetoutput = torch.ones((2,2), requires_grad=True)
input = torch.ones((2,2), requires_grad=True)
kernel = torch.ones((2,2), requires_grad=True)


#ニューラルネットワークからSeparableConvolutionへの出力
output = subnetoutput * 2
#C言語を通過するのでAutoGradが切れる
input = torch.tensor(output.data, requires_grad=True)
#画像出力の計算を行う
out1 = input * kernel

out1 = out1.mean()

out1.backward()

print(input.grad)
#ここではsubnetoutputの勾配を計算することができない
print(subnetoutput.grad)
# d out1 / d subnetoutput =
output.backward(input.grad)
print(subnetoutput.grad)



print("from here ")

#torch.Size([1, 256, 32, 40])
#torch.Size([1, 256, 64, 80])

matrix1 = torch.ones(1,256,32,40)
matrix2 = torch.ones(1,256,64,80)

matrix3 = matrix2 + matrix1

print(matrix3)















#
