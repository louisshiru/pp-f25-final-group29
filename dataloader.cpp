#include "dataloader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Trim helper ensures we can safely match TSPLIB tokens.
std::string trim(const std::string& input) {
	auto is_ws = [](unsigned char ch) { return std::isspace(ch) != 0; };
	auto first = std::find_if_not(input.begin(), input.end(), is_ws);
	if (first == input.end()) return "";
	auto last = std::find_if_not(input.rbegin(), input.rend(), is_ws).base();
	return std::string(first, last);
}

std::string resolve_dataset_path(const std::string& dataset_name) {
	if (dataset_name.empty()) {
		throw std::invalid_argument("Dataset name cannot be empty");
	}

	const bool contains_slash = dataset_name.find('/') != std::string::npos || dataset_name.find('\\') != std::string::npos;
	std::vector<std::string> candidates = {dataset_name};

	if (!contains_slash) {
		const std::string prefixes[] = {"data/", "dataset/", "../data/", "../dataset/"};
		for (const auto& prefix : prefixes) {
			candidates.emplace_back(prefix + dataset_name);
		}
	}

	for (const auto& candidate : candidates) {
		std::ifstream file(candidate);
		if (file.is_open()) {
			return candidate;
		}
	}

	throw std::runtime_error("Unable to locate dataset file: " + dataset_name);
}

std::vector<City> load_tsplib(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open dataset: " + path);
	}

	std::vector<City> parsed_cities;
	std::string line;
	bool in_node_section = false;
	int declared_dimension = -1;

	while (std::getline(file, line)) {
		line = trim(line);
		if (line.empty()) {
			continue;
		}

		if (!in_node_section) {
			if (line == "NODE_COORD_SECTION") {
				in_node_section = true;
				if (declared_dimension > 0) {
					parsed_cities.reserve(declared_dimension);
				}
				continue;
			}

			const auto colon_pos = line.find(':');
			if (colon_pos != std::string::npos) {
				const auto key = trim(line.substr(0, colon_pos));
				const auto value = trim(line.substr(colon_pos + 1));
				if (key == "DIMENSION") {
					declared_dimension = std::stoi(value);
				}
			}
			continue;
		}

		if (line == "EOF") {
			break;
		}

		std::istringstream iss(line);
		City city{};
		if (!(iss >> city.id >> city.x >> city.y)) {
			throw std::runtime_error("Malformed coordinate line in " + path + ": " + line);
		}
		parsed_cities.push_back(city);
	}

	if (parsed_cities.empty()) {
		throw std::runtime_error("Dataset contains no coordinates: " + path);
	}

	if (declared_dimension > 0 && static_cast<int>(parsed_cities.size()) != declared_dimension) {
		throw std::runtime_error(
			"DIMENSION mismatch for " + path + ": expected " + std::to_string(declared_dimension) +
			", parsed " + std::to_string(parsed_cities.size()));
	}

	return parsed_cities;
}

} // namespace

Dataloader::Dataloader(const std::string& dataset_name) {
	const std::string dataset_path = resolve_dataset_path(dataset_name);
	cities = load_tsplib(dataset_path);
}

