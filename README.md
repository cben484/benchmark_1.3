# FIRST OF ALL

this is just for my first startup project of benchmark based cuda

# version
benchmark_1.2

# version show
this version is really for my first completed benchmark without n test .
it works the single and the double validate done .

# attention
attention , the syrk/potrf just store its result in lower/higher part ,
if you just cudamalloc ,only if you are lucky ,you cant get the correct matrix ,
you should init the matrix or assign 0 to α
if you need a warmup for your procedure, dont put the potrf in the for , just put the gemm in it ,
for validate, dont deem the matrix A after potrf is A, the real A is the matrix A before potrf ,
actually , the matrix A after potrf is L .
