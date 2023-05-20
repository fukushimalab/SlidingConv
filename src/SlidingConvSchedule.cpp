#include "SlidingConv.h"

SlidingConvRefSchedule SlidingConv::scheduleX()
{
	return SlidingConvRefSchedule(this->content, true);
}

SlidingConvRefSchedule SlidingConv::scheduleY()
{
	return SlidingConvRefSchedule(this->content, false);
}

void SlidingConv::cpu_auto_schedule()
{
	scheduleX()
		.tile(content.width / 2, 16)
		.parallel({XO, YO, C})
		.reorder({ XI, YI, XO, YO, C })
		.unroll(XI)
		;

	scheduleY()
		.tile(16, content.height / 2)
		.parallel({ XO, YO, C })
		.reorder({ YI, XI, XO, YO, C })
		.vectorize(XI)
		;
}

void SlidingConv::gpu_auto_schedule()
{
	scheduleX()
		.tile(8, 8)
		.gpuBlocks({XO, YO})
		.gpuThreads({YI, XI})
		.reorder({ XI, YI, XO, YO, C })
		;

	scheduleY()
		.tile(8, 8)
		.gpuBlocks({ XO, YO })
		.gpuThreads({ XI, YI })
		.reorder({ YI, XI, XO, YO, C });
}