import sys
data = float(sys.stdin.readline())

if data >= 4.00:
    print(8.00)
else:
    print(round(2*data, 2))
