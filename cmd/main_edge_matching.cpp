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
#include "../include/memory_stats.h"

#if USE_GLOGS
#include <glog/logging.h>
#endif

//> usage: (Under the bin file) ./main_edge_matching --config_file ../config/eth3d_slam.yaml

//> Define default values for the input argument
#if USE_GLOGS
DEFINE_string(config_file, "../config/eth3d_slam.yaml", "config file path");
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

	if (config_map["clear_output_dir"] && config_map["clear_output_dir"].as<bool>()) {
		io_tool->clear_output_directory(dataset_->get_output_path());
	}

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

	//> Write all the stereo GPU timings to a file
	std::string output_dir = dataset_->get_output_path();
#if MEASURE_MEMORY
	quad_vo_sys->sample_run_memory();
#endif
	std::string gpu_timings_filename = output_dir + "/gpu_timings_stereo_matches.txt";
	std::ofstream gpu_timings_file(gpu_timings_filename);
	for (size_t i = 0; i < quad_vo_sys->collect_stereo_GPU_times.size(); i++) {
		gpu_timings_file << quad_vo_sys->collect_stereo_GPU_times[i].time_spatial_grid_mapping << " " 
		<< quad_vo_sys->collect_stereo_GPU_times[i].time_epipolar_rasterization << " " 
		<< quad_vo_sys->collect_stereo_GPU_times[i].time_precompute_left_edge_patches_photometry << " " 
		<< quad_vo_sys->collect_stereo_GPU_times[i].time_apply_NCC_filter << " " 
		<< quad_vo_sys->collect_stereo_GPU_times[i].time_apply_SIFT_descriptor_filter << " " 
		<< quad_vo_sys->collect_stereo_GPU_times[i].time_apply_BNB_NCC_filter << " " 
		<< quad_vo_sys->collect_stereo_GPU_times[i].time_apply_BNB_SIFT_filter << " " 
		<< quad_vo_sys->collect_stereo_GPU_times[i].time_apply_epipolar_shift_photometric_refine_and_merge << " " 
		<< quad_vo_sys->collect_stereo_GPU_times[i].time_apply_second_NCC_filter << " " 
		<< quad_vo_sys->collect_stereo_GPU_times[i].time_apply_best_second_ncc_per_left << " " 
		<< quad_vo_sys->collect_stereo_GPU_times[i].total_time << std::endl;
	}
	gpu_timings_file.close();

	//> Write all the temporal GPU timings to a file
	std::string temporal_gpu_timings_filename = output_dir + "/gpu_timings_temporal_matches.txt";
	std::ofstream temporal_gpu_timings_file(temporal_gpu_timings_filename);
	for (size_t i = 0; i < quad_vo_sys->collect_temporal_GPU_times.size(); i++) {
		temporal_gpu_timings_file << quad_vo_sys->collect_temporal_GPU_times[i].time_build_cf_left_spatial_grid << " "
		<< quad_vo_sys->collect_temporal_GPU_times[i].time_temporal_candidate_generation << " "
		<< quad_vo_sys->collect_temporal_GPU_times[i].time_apply_ncc_filter << " "
		<< quad_vo_sys->collect_temporal_GPU_times[i].time_apply_sift_filter << " "
		<< quad_vo_sys->collect_temporal_GPU_times[i].time_apply_bnb_ncc_filter << " "
		<< quad_vo_sys->collect_temporal_GPU_times[i].time_apply_bnb_sift_filter << " "
		<< quad_vo_sys->collect_temporal_GPU_times[i].time_apply_photometric_refine << " "
		<< quad_vo_sys->collect_temporal_GPU_times[i].time_apply_lowe_ratio_filter << " "
		<< quad_vo_sys->collect_temporal_GPU_times[i].total_time << std::endl;
	}
	temporal_gpu_timings_file.close();

#if MEASURE_MEMORY
	const Memory_Usage_Statistics run_peak = quad_vo_sys->run_peak_memory();
	std::cout << "\n[Memory] Run peak GPU used: "
	          << MemoryMonitor::format_bytes(run_peak.gpu_peak_used_bytes)
	          << " / " << MemoryMonitor::format_bytes(run_peak.gpu_total_bytes)
	          << ", host RSS peak: "
	          << MemoryMonitor::format_bytes(run_peak.host_peak_rss_bytes)
	          << std::endl;

	std::string stereo_mem_filename = output_dir + "/gpu_memory_stereo_matches.txt";
	std::ofstream stereo_mem_file(stereo_mem_filename);
	stereo_mem_file << "# gpu_peak_used_bytes gpu_total_bytes host_peak_rss_bytes\n";
	for (size_t i = 0; i < quad_vo_sys->collect_stereo_GPU_times.size(); i++) {
		const auto& mem = quad_vo_sys->collect_stereo_GPU_times[i].memory;
		stereo_mem_file << mem.gpu_peak_used_bytes << " "
		                << mem.gpu_total_bytes << " "
		                << mem.host_peak_rss_bytes << "\n";
	}
	stereo_mem_file.close();

	std::string temporal_mem_filename = output_dir + "/gpu_memory_temporal_matches.txt";
	std::ofstream temporal_mem_file(temporal_mem_filename);
	temporal_mem_file << "# gpu_peak_used_bytes gpu_total_bytes host_peak_rss_bytes\n";
	for (size_t i = 0; i < quad_vo_sys->collect_temporal_GPU_times.size(); i++) {
		const auto& mem = quad_vo_sys->collect_temporal_GPU_times[i].memory;
		temporal_mem_file << mem.gpu_peak_used_bytes << " "
		                  << mem.gpu_total_bytes << " "
		                  << mem.host_peak_rss_bytes << "\n";
	}
	temporal_mem_file.close();

	std::string run_mem_filename = output_dir + "/run_peak_memory.txt";
	std::ofstream run_mem_file(run_mem_filename);
	run_mem_file << "# gpu_peak_used_bytes gpu_total_bytes host_peak_rss_bytes\n";
	run_mem_file << run_peak.gpu_peak_used_bytes << " "
	             << run_peak.gpu_total_bytes << " "
	             << run_peak.host_peak_rss_bytes << "\n";
	run_mem_file.close();

	std::cout << "[Memory] Wrote per-pass stats to " << stereo_mem_filename
	          << " and " << temporal_mem_filename
	          << "; run peak to " << run_mem_filename << std::endl;
#endif

	return 0;
}