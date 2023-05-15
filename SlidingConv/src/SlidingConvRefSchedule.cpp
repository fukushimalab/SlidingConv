#include "SlidingConv.h"

SlidingConvRefSchedule::SlidingConvRefSchedule(SlidingConvContent& content, bool isX): content(content), isX(isX)
{}

SlidingConvRefSchedule& SlidingConvRefSchedule::tile(int splitX, int splitY)
{
	if (isX)
	{
		content.scheduleX.splitX = splitX;
		content.scheduleX.splitY = splitY;
	}
	else
	{
		content.scheduleY.splitX = splitX;
		content.scheduleY.splitY = splitY;
	}

	return *this;
}

SlidingConvRefSchedule& SlidingConvRefSchedule::tile(int splitXAndY)
{
	return tile(splitXAndY, splitXAndY);
}

SlidingConvRefSchedule& SlidingConvRefSchedule::reorder(std::vector<ScheduleTag> order)
{
	if (isX)
	{
		content.scheduleX.reorder = order;
	}
	else
	{
		content.scheduleY.reorder = order;
	}

	return *this;
}

SlidingConvRefSchedule& SlidingConvRefSchedule::parallel(ScheduleTag tag)
{
	if (isX)
	{
		content.scheduleX.parallel.push_back(tag);
	}
	else
	{
		content.scheduleY.parallel.push_back(tag);
	}

	return *this;
}

SlidingConvRefSchedule& SlidingConvRefSchedule::parallel(std::vector<ScheduleTag> tags)
{
	if (isX)
	{
		for (auto tag : tags)
		{
			content.scheduleX.parallel.push_back(tag);
		}
	}
	else
	{
		for (auto tag : tags)
		{
			content.scheduleY.parallel.push_back(tag);
		}
	}

	return *this;
}

SlidingConvRefSchedule& SlidingConvRefSchedule::vectorize(ScheduleTag tag, int factor)
{
	if (isX)
	{
		content.scheduleX.vectorize.push_back(std::pair<ScheduleTag, int>(tag, factor));
	}
	else
	{
		content.scheduleY.vectorize.push_back(std::pair<ScheduleTag, int>(tag, factor));
	}

	return *this;
}

SlidingConvRefSchedule& SlidingConvRefSchedule::vectorize(ScheduleTag tag)
{
	if (isX)
	{
		content.scheduleX.vectorize.push_back(std::pair<ScheduleTag, int>(tag, 0));
	}
	else
	{
		content.scheduleY.vectorize.push_back(std::pair<ScheduleTag, int>(tag, 0));
	}

	return *this;
}

SlidingConvRefSchedule& SlidingConvRefSchedule::unroll(ScheduleTag tag, int factor)
{
	if (isX)
	{
		content.scheduleX.unroll.push_back(std::pair<ScheduleTag, int>(tag, factor));
	}
	else
	{
		content.scheduleY.unroll.push_back(std::pair<ScheduleTag, int>(tag, factor));
	}

	return *this;
}

SlidingConvRefSchedule& SlidingConvRefSchedule::unroll(ScheduleTag tag)
{
	if (isX)
	{
		content.scheduleX.unroll.push_back(std::pair<ScheduleTag, int>(tag, 0));
	}
	else
	{
		content.scheduleY.unroll.push_back(std::pair<ScheduleTag, int>(tag, 0));
	}

	return *this;
}

SlidingConvRefSchedule& SlidingConvRefSchedule::gpuBlocks(std::vector<ScheduleTag> gpuBlocks)
{
	if (isX)
	{
		content.scheduleX.gpuBlocks = gpuBlocks;
	}
	else
	{
		content.scheduleY.gpuBlocks = gpuBlocks;
	}

	return *this;
}

SlidingConvRefSchedule& SlidingConvRefSchedule::gpuThreads(std::vector<ScheduleTag> gpuThreads)
{
	if (isX)
	{
		content.scheduleX.gpuThreads = gpuThreads;
	}
	else
	{
		content.scheduleY.gpuThreads = gpuThreads;
	}

	return *this;
}