import csv
if __name__=='__main__':
    n=[["lufei","nan",100],["susu","nan",99],["namei","nv",90]]

    with open("IP2.csv",'w',newline='') as t:
        writer=csv.writer(t)
        writer.writerow([100])
        writer.writerows(n)
