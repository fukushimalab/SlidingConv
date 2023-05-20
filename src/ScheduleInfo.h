#pragma once

enum ScheduleTag
{
	XO,
	XI,
	YO,
	YI,
	C
};

struct ScheduleInfo
{
	std::vector<ScheduleTag> parallel;

	std::vector<ScheduleTag> reorder;

	std::vector<ScheduleTag> gpuBlocks;

	std::vector<ScheduleTag> gpuThreads;

	std::vector<std::pair<ScheduleTag, int>> vectorize;

	std::vector<std::pair<ScheduleTag, int>> unroll;

	int splitX;
	int splitY;
};