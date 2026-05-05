#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include "../include/definitions.h"
#include "../include/Dataset.h"
#include "../include/Pipeline.h"
#include "../include/Temporal_Matches.h"
#include "../include/io.h"

#if USE_GLOGS
#include <glog/logging.h>
#endif

//> usage: (Under the bin file) ./main_VO --config_file ../config/eth3d_cable_2.yaml

//> Define default values for the input argument
#if USE_GLOGS
DEFINE_string(config_file, "../config/eth3d_cable_2.yaml", "config file path");
#endif

int main(int argc, char **argv)
{
	//> Get input arguments
#if USE_GLOGS
	google::ParseCommandLineFlags(&argc, &argv, true);
#else
	//> Get input argument
	--argc;
	++argv;
	std::string arg;
	int argIndx = 0, argTotal = 4;
	std::string FLAGS_config_file;

	if (argc)
	{
		arg = std::string(*argv);
		if (arg == "-h" || arg == "--help")
		{
			LOG_PRINT_HELP_MESSAGE;
			return 0;
		}
		else if (argc <= argTotal)
		{
			while (argIndx <= argTotal - 1)
			{
				if (arg == "-c" || arg == "--config_file")
				{
					argv++;
					arg = std::string(*argv);
					FLAGS_config_file = arg;
					argIndx += 2;
					break;
				}
				else
				{
					LOG_ERROR("Invalid input arguments! Follow the instruction:");
					LOG_PRINT_HELP_MESSAGE;
					return 0;
				}
				argv++;
			}
		}
		else if (argc > argTotal)
		{
			LOG_ERROR("Too many input arguments! Follow the instruction:");
			LOG_PRINT_HELP_MESSAGE;
			return 0;
		}
	}
	else
	{
		LOG_PRINT_HELP_MESSAGE;
		return 0;
	}
#endif
	YAML::Node config_map;

	try
	{
		config_map = YAML::LoadFile(FLAGS_config_file);
#if SHOW_YAML_FILE_DATA
		std::cout << config_map << std::endl;
#endif
	}
	catch (const std::exception &e)
	{
		std::cerr << "Exception: " << e.what() << std::endl;
		std::cerr << "File does not exist!" << std::endl;
	}

	//> Initialize the dataset and the pipeline
	Dataset::Ptr dataset_ = std::make_shared<Dataset>(config_map);
	Pipeline::Ptr quad_vo_sys = Pipeline::Ptr(new Pipeline(dataset_));

	//> initialize io class pointer
	quad_Edge_vo_io::Ptr io_tool = nullptr;
	io_tool = std::make_shared<quad_Edge_vo_io>();

	//> Clear the incremental append outputs
	io_tool->reset_incremental_append_outputs();

	//> Intentionally start with the N-th frame (for debugging purpose)
	const int first_stereo_frame_index = 0;
	quad_vo_sys->set_Current_Stereo_Frame_Index(first_stereo_frame_index); 

	//> Loop over all stereo image pairs
	for (int frame_idx = 0; frame_idx < dataset_->get_Total_Num_of_Stereo_Frames(); frame_idx++) {

		if (!dataset_->stereo_iterator->getNext(quad_vo_sys->current_frame, true)) {
            std::cout << "No more stereo image pairs to process" << std::endl;
            break;
        }
		
		quad_vo_sys->Add_Stereo_Frame();

		//> This is only for the debugging purpose
		if (quad_vo_sys->b_end_pipeline_for_debug) {
			break;
		}

		if (quad_vo_sys->get_Status() == PipelineStatus::STATUS_SYSTEM_EXIT) {
			std::cout << "System exit: " << quad_vo_sys->system_exit_message << std::endl;
			break;
		}

		if (quad_vo_sys->get_Status() == PipelineStatus::STATUS_IMG_PREPARATION) {
			//> Fetch the next stereo frame
			continue;
		}
		else { 
			break;
		}
	}

	return 0;
}