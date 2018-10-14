"""
# print model parametrs
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size(), param_tensor)

for param in net.parameters():
    print(param.size())
"""