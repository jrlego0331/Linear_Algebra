import numpy as np
# 11기 20135 이정로, 20168 정진용 Free of Use

row, col = int(input('Row: ')), int(input('Col: '))
matrix = np.zeros([col, row])

for j in range(col):
    for i in range(row):
        output = 'product '+ str(i) + ',' + str(j) + ': '
        matrix[j, i] = float(input(output))

print('orignal matrix: \n', matrix, '\n')

#추축의 1행이 0일경우 마지막 행으로 옮기는 함수
def zeroCheck(grid, starty, startx):
    for j in range(starty, col):
        if grid[j, startx] == 0:
            grid = np.roll(grid, int((col- j-1)* row))
    return grid

#선행 선분(Pivot) 아래에 있는 성분들은 모두 0인지 확인하는 함수
def gausssianCheck(grid, starty, staticx):
    count = 0
    for y in range(starty+1, col):
        if grid[y, 0] == 0:
            count +=1
    
    if count == col -starty -1:
        return(True)
        
    else:
        return(False)
        

matrix = zeroCheck(matrix, 0, 0)
print(matrix, '\n')

current_i = 0
flop = 0

for y in range(col):
    while True:
        for j in range(y, col):
            if matrix[j, current_i] != 0:
                matrix[j, current_i:row] = np.divide(matrix[j, current_i:row],  matrix[j,current_i])
                flop += 1
        print(matrix, '\n')

        for j in range(y+1, col):
            if matrix[j, current_i] != 0:
                matrix[j, current_i:row] -= matrix[y, current_i:row]
                flop += 1
        print(matrix, '\n')


        if gausssianCheck(matrix, y, current_i):
            current_i += 1
            break
        
print('flop', flop)