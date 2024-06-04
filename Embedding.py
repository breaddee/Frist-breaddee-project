import matplotlib.pyplot as plt

his_dict = history.history
loss = his_dict['loss']
val_loss = his['val_loss']

epochs = range(1, len(loss) + 1)
fig = plt.figure(figsize = (10, 5))
