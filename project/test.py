import torch

conv1 = torch.nn.Conv2d(2, 1, (3, 3), bias=True) # [каналы на вход X на выход X размер ядра]
conv2 = torch.nn.Conv2d(1, 1, (1, 1), bias=False) # свёртка 1х1. Если посмотреть на ее веса, там будет ровно одно число.

new_conv = torch.nn.Conv2d(1, 1, 3, bias=False) # Вот эта свёртка объеденит две.
new_conv.weight.data = conv1.weight.data * conv2.weight.data 

# проверим что все сработало
x = torch.randn([1, 2, 6, 6]) # [Размер батча X кол-во каналов X размер по вертикали X размер по горизонтали]
out = conv2(conv1(x))
# new_out = new_conv(x)
print(x)
print(conv1(x), end="------------------------------\n")
print(conv2(conv1(x)))
print(out)