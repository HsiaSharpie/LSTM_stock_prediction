import matplotlib.pyplot as plt


def plot_result(comp, real_stock_price, predicted_price):
    # 紅線表示真實股價
    plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price {}'.format(comp)) 
    # 藍線表示預測股價
    plt.plot(predicted_price, color = 'blue', label = 'Predicted Stock Price {}'.format(comp))  
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()