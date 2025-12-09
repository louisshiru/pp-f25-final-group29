
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

# Data on my environment
CPU: AMD Ryzen 9 9950X3D 16-Core Processor
GPU: NVIDIA GeForce RTX 3060 12GB
RAM: 32GB

## GA - without 2opt (serial)

Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 58649.1 Time: 0.0113095s
Generation 100 Best Distance: 58423.3 Time: 1.01003s
Generation 200 Best Distance: 57021.8 Time: 1.0157s
Generation 300 Best Distance: 56525.5 Time: -0.46106s
Generation 400 Best Distance: 55839.6 Time: 1.00863s
Generation 500 Best Distance: 54691.5 Time: 1.00663s
Generation 600 Best Distance: 54073.3 Time: 1.0072s
Generation 700 Best Distance: 53428.8 Time: 1.00957s
Generation 800 Best Distance: 53334.8 Time: 1.01156s
Generation 900 Best Distance: 52968.5 Time: 1.00762s
Generation 1000 Best Distance: 51898.5 Time: 1.01297s
Generation 1100 Best Distance: 51898.5 Time: 1.00938s
Generation 1200 Best Distance: 51805.5 Time: 1.00971s
Generation 1300 Best Distance: 51517.1 Time: 1.01412s
Generation 1400 Best Distance: 50618 Time: 1.02149s
Generation 1500 Best Distance: 50498.9 Time: 1.02064s
Generation 1600 Best Distance: 50467 Time: 1.01717s
Generation 1700 Best Distance: 50197.2 Time: 1.02019s
Generation 1800 Best Distance: 49724.1 Time: 1.01842s
Generation 1900 Best Distance: 49668.1 Time: 1.02239s
Generation 2000 Best Distance: 47939.6 Time: 1.01873s
Generation 2100 Best Distance: 47844 Time: 1.01972s
Generation 2200 Best Distance: 47838.5 Time: 1.0198s
Generation 2300 Best Distance: 46439.8 Time: 1.0186s
Generation 2400 Best Distance: 46235.5 Time: 1.01506s
Generation 2500 Best Distance: 45950.6 Time: 1.01811s
Generation 2600 Best Distance: 45630.3 Time: 1.01594s
Generation 2700 Best Distance: 45590.4 Time: 1.01805s
Generation 2800 Best Distance: 45533.4 Time: 1.01946s
Generation 2900 Best Distance: 45202.8 Time: 1.01193s
Generation 2999 Best Distance: 45151.4 Time: 1.00217s
Final Best Distance: 45151.4
Time: 28.9716 s

## GA - all(serial - 1pass opt)
Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 60009.3 Time: 0.0135413s
Generation 100 Best Distance: 53036.2 Time: 1.07295s
Generation 200 Best Distance: 49244.9 Time: 1.05894s
Generation 300 Best Distance: 43271.7 Time: 1.05898s
Generation 400 Best Distance: 38338.3 Time: 1.05869s
Generation 500 Best Distance: 34904.2 Time: 1.06032s
Generation 600 Best Distance: 32886.7 Time: 1.06177s
Generation 700 Best Distance: 31090.8 Time: 1.05986s
Generation 800 Best Distance: 29561.5 Time: 1.05852s
Generation 900 Best Distance: 28412.6 Time: 1.06411s
Generation 1000 Best Distance: 25984.1 Time: 1.05872s
Generation 1100 Best Distance: 22125.7 Time: 1.05617s
Generation 1200 Best Distance: 19858.4 Time: 1.0569s
Generation 1300 Best Distance: 17952.6 Time: 1.05887s
Generation 1400 Best Distance: 16476 Time: -0.502423s
Generation 1500 Best Distance: 14167.3 Time: 1.0678s
Generation 1600 Best Distance: 12279.1 Time: 1.06674s
Generation 1700 Best Distance: 10205.6 Time: 1.07291s
Generation 1800 Best Distance: 10205.6 Time: 1.07447s
Generation 1900 Best Distance: 10205.6 Time: 1.0708s
Generation 2000 Best Distance: 10205.6 Time: 1.0679s
Generation 2100 Best Distance: 10205.6 Time: 1.07054s
Generation 2200 Best Distance: 10205.6 Time: 1.07262s
Generation 2300 Best Distance: 10205.6 Time: 1.0687s
Generation 2400 Best Distance: 10205.6 Time: 1.07037s
Generation 2500 Best Distance: 10205.6 Time: 1.0688s
Generation 2600 Best Distance: 10205.6 Time: 1.06938s
Generation 2700 Best Distance: 10205.6 Time: 1.07216s
Generation 2800 Best Distance: 10189.6 Time: 1.06694s
Generation 2900 Best Distance: 10144.7 Time: 1.0709s
Generation 2999 Best Distance: 10144.7 Time: 1.05949s
Final Best Distance: 10144.7
Time: 30.4068 s

## GA - all(serial)
Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 58865.3 Time: 0.0127198s
Generation 100 Best Distance: 51192.1 Time: 1.07668s
Generation 200 Best Distance: 43191.1 Time: 1.08279s
Generation 300 Best Distance: 37402.9 Time: 1.08115s
Generation 400 Best Distance: 31777.7 Time: 1.08548s
Generation 500 Best Distance: 25873.5 Time: 1.08545s
Generation 600 Best Distance: 23583.9 Time: 1.08704s
Generation 700 Best Distance: 17931.6 Time: 1.09131s
Generation 800 Best Distance: 12666.5 Time: 1.10161s
Generation 900 Best Distance: 10056.6 Time: 1.1192s
Generation 1000 Best Distance: 10011.6 Time: 1.13496s
Generation 1100 Best Distance: 10011.6 Time: 1.14232s
Generation 1200 Best Distance: 10011.6 Time: 1.14805s
Generation 1300 Best Distance: 9916.2 Time: 1.1449s
Generation 1400 Best Distance: 9805.93 Time: 1.15975s
Generation 1500 Best Distance: 9805.93 Time: 1.14426s
Generation 1600 Best Distance: 9805.93 Time: 1.14818s
Generation 1700 Best Distance: 9805.93 Time: 1.15262s
Generation 1800 Best Distance: 9805.93 Time: 1.15403s
Generation 1900 Best Distance: 9795.43 Time: 1.14885s
Generation 2000 Best Distance: 9770 Time: 1.14661s
Generation 2100 Best Distance: 9770 Time: 1.15378s
Generation 2200 Best Distance: 9764.34 Time: 1.14348s
Generation 2300 Best Distance: 9764.34 Time: 1.14824s
Generation 2400 Best Distance: 9764.34 Time: 1.14801s
Generation 2500 Best Distance: 9761.9 Time: -0.30636s
Generation 2600 Best Distance: 9761.9 Time: 1.15026s
Generation 2700 Best Distance: 9761.9 Time: 1.14366s
Generation 2800 Best Distance: 9761.9 Time: 1.15419s
Generation 2900 Best Distance: 9761.9 Time: 1.1428s
Generation 2999 Best Distance: 9742.03 Time: 1.13274s
Final Best Distance: 9742.03
Time: 32.4591 s

## GA - without 2opt (cuda)

Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 60369.5 Time: 0.18613s
Generation 100 Best Distance: 66787 Time: 0.527443s
Generation 200 Best Distance: 67186.8 Time: 0.521773s
Generation 300 Best Distance: 68455.2 Time: 0.522799s
Generation 400 Best Distance: 64854.1 Time: 0.519108s
Generation 500 Best Distance: 62152.5 Time: 0.518705s
Generation 600 Best Distance: 60547.9 Time: 0.518895s
Generation 700 Best Distance: 57673.9 Time: 0.522749s
Generation 800 Best Distance: 55901 Time: 0.521044s
Generation 900 Best Distance: 54305.8 Time: 0.517986s
Generation 1000 Best Distance: 50749.7 Time: 0.517521s
Generation 1100 Best Distance: 49362.1 Time: 0.514881s
Generation 1200 Best Distance: 49134.4 Time: 0.515752s
Generation 1300 Best Distance: 48098.9 Time: 0.516739s
Generation 1400 Best Distance: 47402.8 Time: 0.517746s
Generation 1500 Best Distance: 45716.5 Time: 0.516461s
Generation 1600 Best Distance: 45984 Time: 0.515531s
Generation 1700 Best Distance: 46198.1 Time: 0.518235s
Generation 1800 Best Distance: 46666.1 Time: 0.517456s
Generation 1900 Best Distance: 46839.4 Time: 0.516733s
Generation 2000 Best Distance: 46532.2 Time: -0.983999s
Generation 2100 Best Distance: 45638.8 Time: 0.517778s
Generation 2200 Best Distance: 44922.4 Time: 0.518357s
Generation 2300 Best Distance: 45502.4 Time: 0.523506s
Generation 2400 Best Distance: 44279.5 Time: 0.52113s
Generation 2500 Best Distance: 44649.6 Time: 0.520073s
Generation 2600 Best Distance: 43842.8 Time: 0.518534s
Generation 2700 Best Distance: 43740.4 Time: 0.521197s
Generation 2800 Best Distance: 43054.8 Time: 0.522873s
Generation 2900 Best Distance: 42337.2 Time: 0.521496s
Generation 2999 Best Distance: 42461.4 Time: 0.515731s
Final Best Distance: 42461.4
Time: 14.2621 s

## GA - omp-2opt, cuda-default ga

Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 61548.6 Time: 0.180082s
Generation 100 Best Distance: 44007 Time: 0.632498s
Generation 200 Best Distance: 33407.5 Time: 0.613831s
Generation 300 Best Distance: 19771.4 Time: 0.60371s
Generation 400 Best Distance: 10874 Time: 0.672564s
Generation 500 Best Distance: 9931.8 Time: 0.782606s
Generation 600 Best Distance: 9931.8 Time: 0.872452s
Generation 700 Best Distance: 9931.8 Time: 0.756588s
Generation 800 Best Distance: 9931.8 Time: 0.722926s
Generation 900 Best Distance: 9921.45 Time: 0.72207s
Generation 1000 Best Distance: 9931.8 Time: 0.878289s
Generation 1100 Best Distance: 9931.8 Time: 0.739795s
Generation 1200 Best Distance: 9931.8 Time: 0.764325s
Generation 1300 Best Distance: 9931.8 Time: 0.768712s
Generation 1400 Best Distance: 9931.8 Time: 0.803249s
Generation 1500 Best Distance: 9931.8 Time: 0.839435s
Generation 1600 Best Distance: 9931.8 Time: 1.07606s
Generation 1700 Best Distance: 9931.8 Time: 0.75125s
Generation 1800 Best Distance: 9931.8 Time: 0.845364s
Generation 1900 Best Distance: 9931.8 Time: 0.945619s
Generation 2000 Best Distance: 9931.8 Time: 0.812708s
Generation 2100 Best Distance: 9931.8 Time: 0.745177s
Generation 2200 Best Distance: 9931.8 Time: 0.741487s
Generation 2300 Best Distance: 9931.8 Time: 0.800158s
Generation 2400 Best Distance: 9931.8 Time: 0.767041s
Generation 2500 Best Distance: 9931.8 Time: 0.741681s
Generation 2600 Best Distance: 9931.8 Time: 0.719773s
Generation 2700 Best Distance: 9929.18 Time: 0.74332s
Generation 2800 Best Distance: 9931.8 Time: 1.02205s
Generation 2900 Best Distance: 9931.8 Time: 0.92436s
Generation 2999 Best Distance: 9925.84 Time: 0.714896s
Final Best Distance: 9925.84
Time: 23.7067 s

## GA - all (cuda 2pass opt)

Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 62666.8 Time: 0.359478s
Generation 100 Best Distance: 45647.5 Time: 0.734488s
Generation 200 Best Distance: 33436.3 Time: 0.99991s
Generation 300 Best Distance: 18226.1 Time: -0.077482s
Generation 400 Best Distance: 12081.3 Time: 1.94488s
Generation 500 Best Distance: 9919 Time: 2.16074s
Generation 600 Best Distance: 9919 Time: 2.03468s
Generation 700 Best Distance: 9919 Time: 1.99053s
Generation 800 Best Distance: 9919 Time: 1.90088s
Generation 900 Best Distance: 9919 Time: 2.03537s
Generation 1000 Best Distance: 9919 Time: 1.90361s
Generation 1100 Best Distance: 9919 Time: 2.07738s
Generation 1200 Best Distance: 9919 Time: 2.11568s
Generation 1300 Best Distance: 9919 Time: 2.03414s
Generation 1400 Best Distance: 9919 Time: 2.00823s
Generation 1500 Best Distance: 9919 Time: 1.92942s
Generation 1600 Best Distance: 9919 Time: 2.08595s
Generation 1700 Best Distance: 9919 Time: 2.12803s
Generation 1800 Best Distance: 9919 Time: 2.05232s
Generation 1900 Best Distance: 9919 Time: 0.547066s
Generation 2000 Best Distance: 9919 Time: 1.94236s
Generation 2100 Best Distance: 9919 Time: 1.81082s
Generation 2200 Best Distance: 9865.34 Time: 1.52057s
Generation 2300 Best Distance: 9865.34 Time: 2.36609s
Generation 2400 Best Distance: 9865.34 Time: 1.8413s
Generation 2500 Best Distance: 9865.34 Time: 1.97268s
Generation 2600 Best Distance: 9865.34 Time: 2.10027s
Generation 2700 Best Distance: 9865.34 Time: 1.94378s
Generation 2800 Best Distance: 9865.34 Time: 2.0179s
Generation 2900 Best Distance: 9865.34 Time: 1.99654s
Generation 2999 Best Distance: 9865.34 Time: 2.04466s
Final Best Distance: 9865.34
Time: 54.5238 s

## GA - all (cuda 1pass opt)
Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 61806.4 Time: 0.170359s
Generation 100 Best Distance: 56053.5 Time: 0.638494s
Generation 200 Best Distance: 49808.6 Time: 0.633721s
Generation 300 Best Distance: 46227.8 Time: 0.628585s
Generation 400 Best Distance: 37478.1 Time: 0.630856s
Generation 500 Best Distance: 34165.7 Time: 0.66427s
Generation 600 Best Distance: 31030.8 Time: 0.805135s
Generation 700 Best Distance: 23485.3 Time: 0.94823s
Generation 800 Best Distance: 16862.1 Time: 1.10435s
Generation 900 Best Distance: 11593.4 Time: -0.633413s
Generation 1000 Best Distance: 9857.57 Time: 1.16018s
Generation 1100 Best Distance: 9857.57 Time: 1.27215s
Generation 1200 Best Distance: 9857.57 Time: 1.27064s
Generation 1300 Best Distance: 9857.57 Time: 1.26126s
Generation 1400 Best Distance: 9857.57 Time: 1.25746s
Generation 1500 Best Distance: 9854.58 Time: 1.2636s
Generation 1600 Best Distance: 9857.57 Time: 1.27852s
Generation 1700 Best Distance: 9857.57 Time: 1.27818s
Generation 1800 Best Distance: 9857.57 Time: 1.28179s
Generation 1900 Best Distance: 9857.57 Time: 1.27155s
Generation 2000 Best Distance: 9857.57 Time: 1.26914s
Generation 2100 Best Distance: 9857.57 Time: 1.25769s
Generation 2200 Best Distance: 9857.57 Time: 1.26104s
Generation 2300 Best Distance: 9857.57 Time: 1.26949s
Generation 2400 Best Distance: 9857.57 Time: 1.26171s
Generation 2500 Best Distance: 9857.57 Time: 1.29203s
Generation 2600 Best Distance: 9857.57 Time: 1.25978s
Generation 2700 Best Distance: 9857.57 Time: 1.2802s
Generation 2800 Best Distance: 9857.57 Time: 1.27938s
Generation 2900 Best Distance: 9857.57 Time: 1.28748s
Generation 2999 Best Distance: 9857.57 Time: 1.2526s
Final Best Distance: 9857.57
Time: 32.1579 s

# New data on department server

## GA - without 2opt (serial)

Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 60559.9 Time: 0.0279385s
Generation 100 Best Distance: 55974.1 Time: 2.61634s
Generation 200 Best Distance: 55408.3 Time: 2.62914s
Generation 300 Best Distance: 55375.6 Time: 2.61609s
Generation 400 Best Distance: 54656.9 Time: 2.63163s
Generation 500 Best Distance: 54383 Time: 2.61458s
Generation 600 Best Distance: 53108.9 Time: 2.6162s
Generation 700 Best Distance: 52442.8 Time: 2.62069s
Generation 800 Best Distance: 50661.9 Time: 2.63086s
Generation 900 Best Distance: 49812.1 Time: 2.61814s
Generation 1000 Best Distance: 49059.8 Time: 2.61684s
Generation 1100 Best Distance: 48119.6 Time: 2.62708s
Generation 1200 Best Distance: 47691.4 Time: 2.61923s
Generation 1300 Best Distance: 47518.2 Time: 2.62167s
Generation 1400 Best Distance: 47398 Time: 2.62105s
Generation 1500 Best Distance: 47260.1 Time: 2.63055s
Generation 1600 Best Distance: 46810 Time: 2.61882s
Generation 1700 Best Distance: 46568.3 Time: 2.61581s
Generation 1800 Best Distance: 46568.3 Time: 2.62536s
Generation 1900 Best Distance: 46378.5 Time: 2.63746s
Generation 2000 Best Distance: 46259.2 Time: 2.62116s
Generation 2100 Best Distance: 45754.4 Time: 2.61811s
Generation 2200 Best Distance: 45635 Time: 2.61541s
Generation 2300 Best Distance: 45635 Time: 2.62616s
Generation 2400 Best Distance: 45596.8 Time: 2.60854s
Generation 2500 Best Distance: 45596.8 Time: 2.61615s
Generation 2600 Best Distance: 45058.1 Time: 2.61198s
Generation 2700 Best Distance: 44628.2 Time: 2.60502s
Generation 2800 Best Distance: 44052.4 Time: 2.59641s
Generation 2900 Best Distance: 43818.1 Time: 2.59861s
Generation 2999 Best Distance: 42965.9 Time: 2.58431s
Final Best Distance: 42965.9
Time: 78.5578 s

## GA - all(serial - 2pass opt)

Generation 0 Best Distance: 56949.6 Time: 0.0321565s
Generation 100 Best Distance: 49701.9 Time: 2.75542s
Generation 200 Best Distance: 43263.8 Time: 2.77193s
Generation 300 Best Distance: 36657.2 Time: 2.75661s
Generation 400 Best Distance: 32887.7 Time: 2.77231s
Generation 500 Best Distance: 27889.7 Time: 2.76794s
Generation 600 Best Distance: 23736.3 Time: 2.86867s
Generation 700 Best Distance: 18304.6 Time: 2.84624s
Generation 800 Best Distance: 13051.7 Time: 2.93425s
Generation 900 Best Distance: 10506.1 Time: 3.03618s
Generation 1000 Best Distance: 9859.59 Time: 3.11839s
Generation 1100 Best Distance: 9859.59 Time: 3.16374s
Generation 1200 Best Distance: 9859.59 Time: 3.13754s
Generation 1300 Best Distance: 9859.59 Time: 3.15043s
Generation 1400 Best Distance: 9852.33 Time: 3.11474s
Generation 1500 Best Distance: 9846.27 Time: 3.29686s
Generation 1600 Best Distance: 9846.27 Time: 3.15941s
Generation 1700 Best Distance: 9797.01 Time: 3.1171s
Generation 1800 Best Distance: 9797.01 Time: 3.1396s
Generation 1900 Best Distance: 9786.5 Time: 3.16579s
Generation 2000 Best Distance: 9783.94 Time: 3.11957s
Generation 2100 Best Distance: 9783.94 Time: 3.18922s
Generation 2200 Best Distance: 9783.94 Time: 3.1069s
Generation 2300 Best Distance: 9783.94 Time: 3.19564s
Generation 2400 Best Distance: 9779.82 Time: 3.17138s
Generation 2500 Best Distance: 9779.82 Time: 3.22499s
Generation 2600 Best Distance: 9779.82 Time: 3.14749s
Generation 2700 Best Distance: 9779.82 Time: 3.19468s
Generation 2800 Best Distance: 9779.82 Time: 3.20446s
Generation 2900 Best Distance: 9779.82 Time: 3.15441s
Generation 2999 Best Distance: 9758.51 Time: 3.09277s
Time: 91.9072 s

## GA - all(serial - 1pass opt)
Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 56897.5 Time: 0.0301193s
Generation 100 Best Distance: 53427 Time: 2.69161s
Generation 200 Best Distance: 50555.6 Time: 2.71206s
Generation 300 Best Distance: 47204 Time: 2.7005s
Generation 400 Best Distance: 43390.7 Time: 2.69786s
Generation 500 Best Distance: 40125.2 Time: 2.69862s
Generation 600 Best Distance: 36745.9 Time: 2.71489s
Generation 700 Best Distance: 33990.9 Time: 2.69583s
Generation 800 Best Distance: 32334 Time: 2.69726s
Generation 900 Best Distance: 27368.7 Time: 2.70257s
Generation 1000 Best Distance: 22722.4 Time: 2.71693s
Generation 1100 Best Distance: 20180.8 Time: 2.72424s
Generation 1200 Best Distance: 19443 Time: 2.73044s
Generation 1300 Best Distance: 16850.8 Time: 2.77162s
Generation 1400 Best Distance: 14073.5 Time: 2.80124s
Generation 1500 Best Distance: 11480.8 Time: 2.88327s
Generation 1600 Best Distance: 10292.2 Time: 2.90706s
Generation 1700 Best Distance: 10292.2 Time: 2.94388s
Generation 1800 Best Distance: 10292.2 Time: 2.86986s
Generation 1900 Best Distance: 10292.2 Time: 2.89045s
Generation 2000 Best Distance: 10256.1 Time: 2.91698s
Generation 2100 Best Distance: 10256.1 Time: 2.91007s
Generation 2200 Best Distance: 10201.6 Time: 2.97198s
Generation 2300 Best Distance: 10201.6 Time: 2.94022s
Generation 2400 Best Distance: 10201.6 Time: 2.91733s
Generation 2500 Best Distance: 10201.6 Time: 2.90684s
Generation 2600 Best Distance: 10201.6 Time: 2.92305s
Generation 2700 Best Distance: 10201.6 Time: 2.93011s
Generation 2800 Best Distance: 10201.6 Time: 2.90722s
Generation 2900 Best Distance: 10201.6 Time: 2.93141s
Generation 2999 Best Distance: 10201.6 Time: 2.9062s
Final Best Distance: 10201.6
Time: 84.7422 s

## GA - without 2opt (cuda)

Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 61013.4 Time: 0.186079s
Generation 100 Best Distance: 65671.6 Time: 1.02521s
Generation 200 Best Distance: 69224.8 Time: 0.747305s
Generation 300 Best Distance: 67259.6 Time: 0.745173s
Generation 400 Best Distance: 64707.1 Time: 0.740573s
Generation 500 Best Distance: 62168.8 Time: 0.73658s
Generation 600 Best Distance: 60395 Time: 0.740522s
Generation 700 Best Distance: 60956.4 Time: 0.737265s
Generation 800 Best Distance: 59143.4 Time: 0.733954s
Generation 900 Best Distance: 59097.2 Time: 0.746769s
Generation 1000 Best Distance: 57016.5 Time: 0.741267s
Generation 1100 Best Distance: 55121.7 Time: 0.726456s
Generation 1200 Best Distance: 54046.1 Time: 0.733297s
Generation 1300 Best Distance: 53485.6 Time: 0.736904s
Generation 1400 Best Distance: 52769.8 Time: 0.734935s
Generation 1500 Best Distance: 53134.1 Time: 0.737051s
Generation 1600 Best Distance: 50898.5 Time: 0.734591s
Generation 1700 Best Distance: 52044.8 Time: 0.736981s
Generation 1800 Best Distance: 49821 Time: 0.729532s
Generation 1900 Best Distance: 47634.4 Time: 0.726247s
Generation 2000 Best Distance: 48147.3 Time: 0.723492s
Generation 2100 Best Distance: 47866.7 Time: 0.721016s
Generation 2200 Best Distance: 48071.7 Time: 0.722485s
Generation 2300 Best Distance: 47325.2 Time: 0.73922s
Generation 2400 Best Distance: 47322.2 Time: 0.72695s
Generation 2500 Best Distance: 47907.3 Time: 0.726528s
Generation 2600 Best Distance: 47945.1 Time: 0.731565s
Generation 2700 Best Distance: 47446.4 Time: 0.72615s
Generation 2800 Best Distance: 46943.8 Time: 0.726987s
Generation 2900 Best Distance: 46975.9 Time: 0.725636s
Generation 2999 Best Distance: 46138.4 Time: 0.717003s
Time: 22.4649 s

## GA - omp-2opt, cuda-default ga

Generation 0 Best Distance: 62156 Time: 0.226406s
Generation 100 Best Distance: 45042.9 Time: 2.2797s
Generation 200 Best Distance: 26219.4 Time: 2.85156s
Generation 300 Best Distance: 20728 Time: 4.28308s
Generation 400 Best Distance: 10899.4 Time: 7.15269s
Generation 500 Best Distance: 10408 Time: 14.4258s
Generation 600 Best Distance: 10408 Time: 12.2999s
Generation 700 Best Distance: 10408 Time: 12.2892s
Generation 800 Best Distance: 10408 Time: 11.892s
Generation 900 Best Distance: 10380.7 Time: 12.4052s
Generation 1000 Best Distance: 10408 Time: 13.5143s
Generation 1100 Best Distance: 10408 Time: 12.5779s
Generation 1200 Best Distance: 10408 Time: 13.3499s
Generation 1300 Best Distance: 10408 Time: 12.2366s
Generation 1400 Best Distance: 10408 Time: 12.2578s
Generation 1500 Best Distance: 10408 Time: 12.503s
Generation 1600 Best Distance: 10408 Time: 12.6227s
Generation 1700 Best Distance: 10408 Time: 12.9523s
Generation 1800 Best Distance: 10408 Time: 12.3939s
Generation 1900 Best Distance: 10408 Time: 13.4364s
Generation 2000 Best Distance: 10408 Time: 11.3761s
Generation 2100 Best Distance: 10408 Time: 12.1865s
Generation 2200 Best Distance: 10408 Time: 12.3322s
Generation 2300 Best Distance: 10408 Time: 12.4864s
Generation 2400 Best Distance: 10408 Time: 12.5168s
Generation 2500 Best Distance: 10408 Time: 11.9726s
Generation 2600 Best Distance: 10408 Time: 12.1619s
Generation 2700 Best Distance: 10408 Time: 12.4472s
Generation 2800 Best Distance: 10408 Time: 13.3706s
Generation 2900 Best Distance: 10408 Time: 11.147s
Generation 2999 Best Distance: 10408 Time: 11.8749s
Final Best Distance: 10408
Time: 341.824 s



## GA - all (cuda 2pass opt)
Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 60620.1 Time: 0.231019s
Generation 100 Best Distance: 46466.6 Time: 1.30754s
Generation 200 Best Distance: 31282.1 Time: 1.57805s
Generation 300 Best Distance: 18377.8 Time: 2.29582s
Generation 400 Best Distance: 12926.8 Time: 2.88139s
Generation 500 Best Distance: 9928.38 Time: 3.99911s
Generation 600 Best Distance: 9928.38 Time: 3.78845s
Generation 700 Best Distance: 9928.38 Time: 3.54511s
Generation 800 Best Distance: 9928.38 Time: 3.78583s
Generation 900 Best Distance: 9928.38 Time: 4.09457s
Generation 1000 Best Distance: 9928.38 Time: 3.35349s
Generation 1100 Best Distance: 9928.38 Time: 3.4265s
Generation 1200 Best Distance: 9928.38 Time: 3.26547s
Generation 1300 Best Distance: 9928.38 Time: 3.64244s
Generation 1400 Best Distance: 9928.22 Time: 4.20377s
Generation 1500 Best Distance: 9928.38 Time: 3.74403s
Generation 1600 Best Distance: 9928.38 Time: 3.66552s
Generation 1700 Best Distance: 9928.38 Time: 4.03959s
Generation 1800 Best Distance: 9928.38 Time: 3.49462s
Generation 1900 Best Distance: 9928.38 Time: 3.41801s
Generation 2000 Best Distance: 9928.38 Time: 3.58607s
Generation 2100 Best Distance: 9920.47 Time: 3.19117s
Generation 2200 Best Distance: 9928.38 Time: 3.49177s
Generation 2300 Best Distance: 9928.38 Time: 3.42037s
Generation 2400 Best Distance: 9928.38 Time: 3.6116s
Generation 2500 Best Distance: 9928.38 Time: 3.41576s
Generation 2600 Best Distance: 9928.38 Time: 3.80831s
Generation 2700 Best Distance: 9928.38 Time: 3.55844s
Generation 2800 Best Distance: 9928.38 Time: 3.40399s
Generation 2900 Best Distance: 9928.38 Time: 3.29727s
Generation 2999 Best Distance: 9928.38 Time: 3.29004s
Final Best Distance: 9928.38
Time: 101.836 s

## GA - all (cuda 1pass opt)
Starting GA + 2-opt for TSP (qa194.tsp)...
Generation 0 Best Distance: 61484.1 Time: 0.458778s
Generation 100 Best Distance: 58002.7 Time: 1.40557s
Generation 200 Best Distance: 49628.2 Time: 1.17707s
Generation 300 Best Distance: 40245.9 Time: 1.24035s
Generation 400 Best Distance: 32944.2 Time: 1.28287s
Generation 500 Best Distance: 25933.7 Time: 1.44438s
Generation 600 Best Distance: 18337.7 Time: 1.51881s
Generation 700 Best Distance: 13971.3 Time: 1.69342s
Generation 800 Best Distance: 11651 Time: 2.06185s
Generation 900 Best Distance: 9965.29 Time: 2.39192s
Generation 1000 Best Distance: 9965.29 Time: 2.61485s
Generation 1100 Best Distance: 9965.29 Time: 2.48154s
Generation 1200 Best Distance: 9965.29 Time: 2.56477s
Generation 1300 Best Distance: 9965.29 Time: 2.3854s
Generation 1400 Best Distance: 9965.29 Time: 2.64797s
Generation 1500 Best Distance: 9965.29 Time: 2.30159s
Generation 1600 Best Distance: 9958.01 Time: 2.59702s
Generation 1700 Best Distance: 9965.29 Time: 2.67633s
Generation 1800 Best Distance: 9965.29 Time: 2.25805s
Generation 1900 Best Distance: 9965.29 Time: 2.45486s
Generation 2000 Best Distance: 9965.29 Time: 2.5629s
Generation 2100 Best Distance: 9965.29 Time: 2.24415s
Generation 2200 Best Distance: 9965.29 Time: 3.27258s
Generation 2300 Best Distance: 9965.29 Time: 2.38894s
Generation 2400 Best Distance: 9965.29 Time: 2.58965s
Generation 2500 Best Distance: 9965.29 Time: 2.53335s
Generation 2600 Best Distance: 9965.29 Time: 3.01977s
Generation 2700 Best Distance: 9965.29 Time: 2.41702s
Generation 2800 Best Distance: 9965.29 Time: 2.70383s
Generation 2900 Best Distance: 9965.29 Time: 2.33674s
Generation 2999 Best Distance: 9965.29 Time: 2.53935s
Final Best Distance: 9965.29
Time: 68.2668 s