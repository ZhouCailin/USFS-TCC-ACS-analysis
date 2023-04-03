import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

workdir = os.path.dirname(os.path.abspath(__file__))

regions_dict = {"Northeast": ['09', '10', '23', '24', '25', '33', '34', '36', '42', '44', '50'],
                "Upper Mideast": ['19', '26', '27', '55'],
                "Ohio Valley": ['17', '18', '21', '29', '39', '47', '54'],
                "Southeast": ['01', '12', '13', '37', '45', '51'],
                "Northern Rockies and Plains": ['30', '31', '38', '46', '56'],
                "South": ['05', '20', '22', '28', '40', '48'],
                "Southwest":['04', '08', '35', '49'],
                "Northwest": ['16', '41', '53'],
                "West": ['06', '32']}

def kde():
    for zonename in regions_dict.keys():

        # zonename = "Northeast"
        # zonename = "Northern Rockies and Plains"
        # 读取CSV文件
        # df = pd.read_csv(workdir + '\\utcNortheast.csv', index_col=0)
        df = pd.read_csv(workdir + '\\utc' + zonename + '.csv', index_col=0)

        var1 = "PovRatioOver2"
        var2 = "UnderPov"

        # 选择指定行数据
        row_white = df.loc[var1]
        row_color = df.loc[var2]

        # 使用Seaborn库中的kdeplot函数生成KDE图
        data = pd.concat([row_white, row_color], axis=1)
        data.columns = [var1, var2]

        # 绘制KDE图
        sns.kdeplot(data=data, shade=True)
        plt.title("KDE Plot of " + var1 + " and " + var2 + " in " + zonename)
        plt.show()

def bart(df, zone, varlist):
    # Select the specified zone column
    zone_df = df[zone]

    # Select the specified variable columns
    var_df = zone_df[varlist].to_frame()

    ax = var_df.plot.bar(width=0.8, align='edge', legend=False)
    for i, col in enumerate(var_df.columns):
        for j, value in enumerate(var_df[col]):
            ax.text(i + j / len(var_df.columns), value + 0.01, f'{value:.2f}', ha='center', fontsize=10)

    ax.set_ylabel("TCC")
    plt.title("Mean TCC of different income ranges in " + zone)
    plt.show()

def bart2(df, zone, varlist):
    zone_df = df[zone]

    # Select the specified variable columns
    data = zone_df[varlist]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(data)), data.values)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)
    ax.set_xlabel('Value')
    ax.set_title('Bar Chart')

    # Add value marks on each bar
    for i, bar in enumerate(bars):
        ax.text(bar.get_width(), i, bar.get_width(), ha='left', va='center')

    plt.show()

if __name__ == "__main__":
    csv = pd.read_csv(workdir + '\\utctract_wei_zone_result.csv', index_col=0, header=0)
    # kde()
    pov_list = ["HHInc0","HHInc10","HHInc15","HHInc25","HHInc35","HHInc50","HHInc75","HHInc100","HHInc150","HHInc200"]
    for zone in regions_dict.keys():
        bart(csv, zone, pov_list)