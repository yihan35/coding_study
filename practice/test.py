# 岛屿数量 形状相同
def numIslands(grid):
    if not grid or not grid[0]:
        return 0
    rows = len(grid)
    # print(rows)
    cols = len(grid[0])
    unique_islands = set()
    def dfs(r,c,current_islands):
        if r < 0 or r>=rows or c<0 or c>=cols or grid[r][c] ==0:
            return 
        grid[r][c] = 0
        current_islands.append((r,c))
        dfs(r+1,c,current_islands)
        dfs(r-1,c,current_islands)
        dfs(r,c+1,current_islands)
        dfs(r,c-1,current_islands)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                island = []
                dfs(r,c,island)
                min_r = min(p[0] for p in island)
                min_c = min(p[1] for p in island)
                normalized = tuple((x-min_r,y-min_c) for x,y in island)
                unique_islands.add(normalized)
    return len(unique_islands)
grid = [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]
print(numIslands(grid))