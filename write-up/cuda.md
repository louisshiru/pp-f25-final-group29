
# Cuda

我們的 GA + local serach 算法中總共有以下幾個關鍵函數可以做加速

1. fitness
2. crossover
3. mutate
4. local search

每個 Generation 世代中會計算一次 fitness 並選擇 k 個菁英留到下一代，剩餘不足種群數 N 的由上一代的種群互相交配、或獨自突變。
流程為 crossover -> mutate -> local search 或是 2. mutate -> local search 兩種，其中又以 local search 最消耗時間，其計算複雜度為 PN^2，其中 P 代表 max_pass。
另外這個流程會需要做 N/2 次，也就是說計算複雜度為 PN^3，看起來很值得加速，尤其是在種群數目高時。

我們設計了 3 + 1 組實驗，有 3 個 Cuda 版本實作 1 個 Serial 版本的實作，在實驗中 N = 8192, k = 1, P = 50

## Serial GA 

實驗結果如下，每 100 Generation 消耗時間為 44.5729s

## ga2opt_run_cuda

該方法把 `2-opt, mutate, cross-over` 都放入運算，並每個 Generation 都將結果拉回 Host 做最小距離的同步。
實驗結果如下，每 100 Generation 消耗時間為 30.9567s

## ga2opt_run_cuda2

該方法只把 **bottleneck: 2-opt**，即 Local search 的部分放入 GPU 進行運算
實驗結果如下，每 100 Generation 消耗時間為 25.4739s

## ga2opt_run_cuda3

該方法為第一種方法的延伸，並將所有計算都丟給 GPU，並在每 100 個 Generation 才拉回作一次輸出。
實驗結果如下，每 100 Generation 消耗時間為 12.3642s (dj38.tsp)

Another test: zi929.tsp

[GPU-GA+2opt] Generation 0 Best Distance: 2.09514e+06 Time: 0.028151s
[GPU-GA+2opt] Generation 1000 Best Distance: 1.22046e+06 Time: 11.4723s
[GPU-GA+2opt] Generation 2000 Best Distance: 944770 Time: 14.5775s
[GPU-GA+2opt] Generation 3000 Best Distance: 746248 Time: 12.8746s
[GPU-GA+2opt] Generation 4000 Best Distance: 588585 Time: 16.3644s
[GPU-GA+2opt] Generation 5000 Best Distance: 508610 Time: 16.2753s
[GPU-GA+2opt] Generation 6000 Best Distance: 416430 Time: 119.464s
[GPU-GA+2opt] Generation 7000 Best Distance: 346338 Time: 86.5836s
[GPU-GA+2opt] Generation 8000 Best Distance: 300283 Time: 79.041s
[GPU-GA+2opt] Generation 9000 Best Distance: 255991 Time: 231.566s
[GPU-GA+2opt] Generation 10000 Best Distance: 210067 Time: 236.518s
[GPU-GA+2opt] Generation 11000 Best Distance: 158113 Time: 645.581s
[GPU-GA+2opt] Generation 12000 Best Distance: 149812 Time: 422.027s
[GPU-GA+2opt] Generation 13000 Best Distance: 133711 Time: 736.75s
[GPU-GA+2opt] Generation 14000 Best Distance: 124535 Time: 1040.01s
[GPU-GA+2opt] Generation 15000 Best Distance: 118094 Time: 1201.51s

sum = 0.028151 + 11.4723 + 14.5775 + 12.8746 + 16.3644 + 16.2753 + 119.464 + 86.5836 + 79.041 + 231.566 + 236.518 + 645.581 + 422.027 + 736.75 + 1040.01 + 1201.51

Another test: qa194.tsp
[GPU-GA+2opt] Generation 0 Best Distance: 61604.2 Time: 0.0246328s
[GPU-GA+2opt] Generation 1000 Best Distance: 19499.1 Time: 12.9456s
[GPU-GA+2opt] Generation 2000 Best Distance: 10148 Time: 82.2898s
[GPU-GA+2opt] Generation 3000 Best Distance: 9732.35 Time: 84.5267s
[GPU-GA+2opt] Generation 4000 Best Distance: 9706.13 Time: 85.5092s

sum = 0.0246328 + 12.9456 + 82.2898 + 84.5267 + 85.5092

## ga2opt_run_cuda4 (without 2-opt)

考慮到 Local search 對於 GPU 的 thread 來說非常耗時間，

## ga2opt_run_cuda5 (ga all in gpu)

Dataset = qa194.tsp

當種群為 32768 時，Serial 每 100 需要 128 sec，共 128 * 10 = 1280

Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 59409.2 Time: 0.100837s
Generation 100 Best Distance: 47014.6 Time: 8.93229s
Generation 200 Best Distance: 41666.3 Time: 8.87751s
Generation 300 Best Distance: 38646.1 Time: 8.90054s
Generation 400 Best Distance: 34091.9 Time: 8.99185s
Generation 500 Best Distance: 32162.5 Time: 9.01161s
Generation 600 Best Distance: 28499.7 Time: 9.08024s
Generation 700 Best Distance: 21762.1 Time: 9.17584s
Generation 800 Best Distance: 18269.3 Time: 9.27337s
Generation 900 Best Distance: 15683.4 Time: 9.4359s
Generation 999 Best Distance: 12385.8 Time: 9.45099s
Final Best Distance: 12385.8
Best Route: 178 185 186 182 173 172 174 183 176 180 177 179 184 192 187 188 190 191 189 193 181 175 171 168 163 162 160 155 144 139 129 126 124 125 131 133 136 141 148 145 137 138 153 149 143 140 146 150 154 135 134 128 130 121 118 112 108 113 110 103 100 98 88 89 93 97 85 84 64 19 62 35 58 61 81 79 86 101 102 90 77 73 68 71 74 75 70 24 22 12 15 7 5 0 3 1 2 6 10 13 16 25 23 20 17 27 32 56 59 50 36 11 8 40 65 66 60 55 47 37 42 43 14 29 34 46 52 48 49 57 39 33 26 54 31 41 4 9 18 44 30 72 53 45 21 28 67 63 69 76 80 83 78 51 38 82 95 92 87 94 91 96 104 116 117 106 111 114 107 99 109 115 105 119 122 120 123 127 132 142 147 159 165 170 169 167 166 161 157 158 164 151 152 156 178
Time: 91.2311 s

speedup = 1280 / 261 = 4.9

當種群為 8192 時，Serial 每 100 需要 128 sec，共 128 * 10 = 91.23

Starting GA + 2-opt for TSP (qa194.tsp)...
Running on GPU: NVIDIA GeForce RTX 3060
Max threads block: 1024
Generation 0 Best Distance: 59527.8 Time: 0.021036s
Generation 100 Best Distance: 44688.4 Time: 1.65337s
Generation 200 Best Distance: 33349.1 Time: 3s
Generation 300 Best Distance: 25032.9 Time: 4.2372s
Generation 400 Best Distance: 19674.4 Time: 5.75242s
Generation 500 Best Distance: 15163.1 Time: 7.26709s
Generation 600 Best Distance: 12024.6 Time: 8.77573s
Generation 700 Best Distance: 9981.76 Time: 10.0977s
Generation 800 Best Distance: 9597.05 Time: 11.2416s
Generation 900 Best Distance: 9508.43 Time: 11.4392s
Generation 999 Best Distance: 9508.43 Time: 11.2414s
Final Best Distance: 9508.43
Best Route: 88 98 100 103 110 129 126 124 125 131 133 139 144 155 160 162 163 148 145 141 136 137 138 153 156 152 149 143 140 151 146 150 154 157 158 164 167 177 180 176 174 172 173 178 171 168 175 181 193 189 186 185 182 183 188 191 190 187 192 184 179 169 166 161 170 165 159 147 142 135 130 128 134 132 127 123 122 119 120 116 115 114 111 109 99 107 106 104 105 117 121 118 113 112 108 101 102 90 92 95 94 96 91 87 82 78 80 83 76 69 63 67 65 72 66 60 57 55 52 51 47 45 53 54 48 49 41 43 34 37 40 42 39 33 46 50 38 36 26 30 31 29 18 14 11 9 8 4 2 1 3 0 5 7 15 12 22 24 13 10 6 16 25 23 20 17 21 28 27 44 56 32 59 68 73 71 77 74 75 86 79 70 81 61 58 35 62 19 64 84 85 97 93 89 88
Time: 75.1314 s

speedup = 91.23 / 75.1314 = 1.214

## Conclusion

隨著種群數目提升，GPU 的效果將越明顯，但種群數目小時，Serial 版本反而會比 GPU 版本快上許多。由工具查看後發現 Bottleneck 都在 cudaDeviceSynchronize，也就是說，每個世代要同步時將會消耗非常多的時間等待記憶體的搬移，為了減少這種搬移，我將 GA 調整成每 100 個 Generation 才從 GPU 同步結果，也就是第三種平行化方法的原因。因為 fitness、crossover、mutate 等運算都會在不同問題有不一樣的實現，且要移植就是要全部移到功能一次移動到 GPU 上才能得到最好的效果，所以整體來說 GA 並不是很簡單能夠移植到 GPU 的演算法，使用 OpenMP、MPI、Threads 等加速方法會更容易得到好的效果。

考慮到 GA 並不像是傳統 CNN 那種類型，一張圖近來，每個 thread 在各個點上作自己的事情，後面可以合併。GA 是一個種群近來，每個種群都是作一樣多的工作，也就是說，workload 在 CUDA + GA 情境下並沒有變少，這是導致 CUDA 真正原因。

一開始將 GA 整個流程都放入 GPU 執行，發現會有太多的 branch, 且 2-opt 執行時間有很大的落差，無法得到好的 performance
後來將 GA 流程分離後各做 kernel 後有不錯的成績。