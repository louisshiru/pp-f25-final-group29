
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
實驗結果如下，每 100 Generation 消耗時間為 12.3642s


## Conclusion

隨著種群數目提升，GPU 的效果將越明顯，但種群數目小時，Serial 版本反而會比 GPU 版本快上許多。由工具查看後發現 Bottleneck 都在 cudaDeviceSynchronize，也就是說，每個世代要同步時將會消耗非常多的時間等待記憶體的搬移，為了減少這種搬移，我將 GA 調整成每 100 個 Generation 才從 GPU 同步結果，也就是第三種平行化方法的原因。因為 fitness、crossover、mutate 等運算都會在不同問題有不一樣的實現，且要移植就是要全部移到功能一次移動到 GPU 上才能得到最好的效果，所以整體來說 GA 並不是很簡單能夠移植到 GPU 的演算法，使用 OpenMP、MPI、Threads 等加速方法會更容易得到好的效果。