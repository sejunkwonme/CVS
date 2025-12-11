#pragma once
#ifdef __INTELLISENSE__
// Intellisense 컴파일러에 보여줄 부분(공백으로 정의)

//KERNEL_ARG2(grid, block) : <<< grid, block >>>
#define KERNEL_ARG2(grid, block)
//KERNEL_ARG3(grid, block, sh_mem) : <<< grid, block, sh_mem >>>
#define KERNEL_ARG3(grid, block, sh_mem)
//KERNEL_ARG4(grid, block, sh_mem, stream) : <<< grid, block, sh_mem, stream >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream)

#else
//실제 코드 컴파일시에 적용되는 부분

#define KERNEL_ARG2(grid, block) <<< grid, block >>>
#define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>

#endif
