# visualization_class.py - Class for processing and visualizing Pythia events.
# Copyright (C) 2024 Torbjorn Sjostrand.
# Authors: Phil Ilten, Ahmed Youssef, Jure Zupan
# # This software is licensed under the GNU GPL v2 or later. See COPYING for
# details. Please respect the MCnet Guidelines, see GUIDELINES for details.

# Keywords: Pythia8; Monte Carlo simulation; event processing; visualization;

# This module defines the `Visualization` class, responsible for processing
# and visualizing particle interactions from Pythia-generated events. It
# provides utilities for:
# - Parsing event data
# - Assigning colors to particles based on categories
# - Handling beam remnants, MPI interactions, and color connections
# - Generating structured JSON output

# Ensure Pythia 8 is properly configured before using this module.


import sys
import pythia8mc as pythia8
import json
import os
import numpy as np
import argparse
from status_meaning import status_meaning
import webbrowser
import datetime


class Visualization:
    """
    Manages the processing and visualization of particle physics events
    using Pythia.This class handles each step of the event processing
    workflow, from initial particle selection to final output preparation,
    based on provided settings.
    """

    def __init__(self, pythia, settings, file_name):
        """
        Initialize Visualization with a Pythia instance, configuration
        settings, and an output file name.

        Args:
            pythia (pythia8.Pythia): A configured instance of Pythia.
            settings (dict): Visualization configuration settings.
            file_name (str): Name of the file for output data.
        """
        self.pythia = pythia
        self.settings = settings
        self.particle_mother_mapping = {}
        self.list_descendants_with_depth = []
        self.particle_positions = {}
        self.rescaled_positions = {}
        self.category_furthest_dist = {}
        self.color_neutral_objects = []
        self.file_name = file_name
        self.highlight_category = "color_connection"

        self.track_color_connection_info = []

        self.color_map = {
            "hard_process": "FF0000",  # Red
            "beam_remnants": "808080",  # Grey
            "MPI": "FF00FF",  # Magenta
            "parton_shower": "008000",  # Green
            "hadronization": "0000FF",  # Blue
            "default": "000000",  # Black for undefined categories
        }

        self.json_data = {
            "Test, tracks playground": {
                "MissingEnergy": {"TestMissingEnergy": []},
                "Tracks": {"TestTracks": []},
                "Vertices": {"TestVertices": []},
                "event number": 999,
                "run number": 999,
            }
        }

    ######## Helper methods ########

    def get_category(self, status, depth=None):
        """
        Determine the category of a particle from its status code.

        Args:
            status (int): The particle's status code.
            depth (Optional[int]): Reserved for future use.

        Returns:
            str or None: The category name if matched; otherwise, None.
        """
        abs_status = abs(status)
        if abs_status in [23, 24]:
            return "hard_process"
        elif abs_status == 63:
            return "beam_remnants"
        elif abs_status in [33, 43]:
            return "MPI"
        elif 41 <= abs_status <= 62 or 64 <= abs_status <= 79:
            return "parton_shower"
        elif 80 <= abs_status <= 89:
            return "hadronization"
        return None  # Not in any defined category

    def assign_color(self, category):
        """
        Assign a hexadecimal color code based on the particle's category.

        Args:
            category (str): The particle category.

        Returns:
            str: The hex color code corresponding to the category.
        """
        return self.color_map.get(category, self.color_map["default"])

    def lighten_color(self, hex_color, factor=0.5):
        """
        Lighten the given hexadecimal color by mixing it with white.

        Args:
            hex_color (str): A hex color code (e.g., "FF0000").
            factor (float): How much to lighten the color. 0.0 returns
                            the original color, while 1.0 returns white.
        Returns:
            str: A lighter hex color code.
        """
        # Convert hex to integer RGB values.
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Compute new RGB values by moving them closer to 255 (white).
        new_r = int(r + (255 - r) * factor)
        new_g = int(g + (255 - g) * factor)
        new_b = int(b + (255 - b) * factor)

        # Format back to hexadecimal.
        return "{:02X}{:02X}{:02X}".format(new_r, new_g, new_b)

    def compute_boost(self):
        """
        Compute the event boost vector as defined by the settings.

        Returns:
            pythia8.Vec4: The computed boost vector.
        """
        boost_mode = self.settings.get("boost_mode", None)
        boost = pythia8.Vec4(0, 0, 0, 0)  # Initialize the boost as zero

        if boost_mode == "cm_incoming":
            for prt in self.pythia.event:
                if abs(prt.status()) == 21:
                    boost += prt.p()

        elif boost_mode == "cm_outgoing":
            for prt in self.pythia.event:
                if abs(prt.status()) in [23, 24, 63]:
                    boost += prt.p()

        return boost

    def generate_unique_colors(self, num_colors):
        """
        Generate a list of unique colors excluding black, red, green, and blue.

        Parameters:
            num_colors: Number of unique colors to generate.

        Returns:
            colors: A list of unique color hex codes.
        """
        # Predefined list of colors excluding black (#000000), red (#FF0000),
        # green (#00FF00), and blue (#0000FF), magenta (FF00FF)
        excluded_colors = {"000000", "FF0000", "00FF00", "0000FF"}

        # Bright colors first
        color_pool = [
            #    "FFFF00",  # Bright Yellow
            "00FFFF",  # Bright Cyan
            "FFA500",  # Bright Orange
            "FF1493",  # Bright Pink
            "ADFF2F",  # Bright Green-Yellow
            "FFD700",  # Bright Gold
            "00CED1",  # Dark Turquoise
            "8A2BE2",  # BlueViolet
            "DAA520",  # GoldenRod
            "4B0082",  # Indigo
            "800080",  # Purple
        ]

        # Make sure we have enough colors; otherwise repeat them cyclically
        colors = [color for color in color_pool if color not in excluded_colors]
        if len(colors) < num_colors:
            # Repeat colors cyclically if needed
            colors = (colors * (num_colors // len(colors) + 1))[:num_colors]

        return colors

    ########## Helper Methods Block1 ##########
    def _process_daughters(self, start, end, next_depth, mother_idx):
        """
        Helper method to process daughter particles.

        Args:
            start (int): The starting daughter index.
            end (int): The ending daughter index.
            next_depth (int): The depth to use for these daughters.
            mother_idx (int): The index of the mother particle.

        Returns:
            List[Tuple[int, int]]: A list of (prt_idx, depth) tuples
            collected from processing the daughter particles.
        """
        descendants = []
        if start > 0:
            if end >= start:
                for i in range(start, end + 1):
                    desc, self.particle_mother_mapping = self.get_descendants(
                        i, next_depth, mother_idx
                    )
                    descendants += desc
            else:
                desc, self.particle_mother_mapping = self.get_descendants(
                    start, next_depth, mother_idx
                )
                descendants += desc
        return descendants

    ########## Helper Methods Block2 ##########
    def has_beam_remnant_ancestor(self, prt_idx):
        """
        Recursively checks if a particle or any of its ancestors is a
        beam remnant.

        Parameters:
            prt_idx: Index of the particle to check.
            particle_mother_mapping: Dictionary mapping particles to
                                     their mothers.
            beam_remnant_indices: Set of particle indices identified
                                     as beam remnants.

        Returns:
            True if the particle or any of its ancestors is a beam remnant,
            False otherwise.
        """
        particle = self.pythia.event[prt_idx]
        particle_status = abs(particle.status())
        if self.get_category(particle_status) == "beam_remnants":
            return True

        if prt_idx not in self.particle_mother_mapping:
            return False

        # Check all mothers recursively
        for mother_index in self.particle_mother_mapping[prt_idx]:
            if self.has_beam_remnant_ancestor(mother_index):
                return True

        return False

    def compute_mid_end_pos(
        self,
        p,
        start_position,
        category=None,
        scale_factor=None,
        rescale=False,
        remnant_key="non_beam_remnant",
    ):
        """
        Estimate the end and middle positions of a particle based on its
        momentum direction.

        Args:
            prt: The particle object from Pythia.
            p: The particle's four-momentum.
            start_position (numpy.ndarray): The starting position vector.
            category (str, optional): Category of the particle.
            scale_factor (float, optional): Manual scaling factor; if None,
                the value from self.settings is used.

        Returns:
            tuple: A tuple (end_position, middle_position) where:
                - end_position (numpy.ndarray): The estimated final position.
                - middle_position (numpy.ndarray): The midpoint between the
                  start and final positions.
        """
        # For MPI particles with centered settings, use a predefined direction
        if category == "MPI" and self.settings.get("mpi_centered", False):
            print("I am MPI")
            direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
            if p.px() < 0:
                direction *= -1
        else:
            momentum_vector = np.array([p.px(), p.py(), p.pz()])
            print(f"momentum_vector: {momentum_vector}")
            norm = np.linalg.norm(momentum_vector)
            if norm > 0:
                direction = momentum_vector / norm
            else:
                print("using default direction")
                direction = np.array([0.0, 0.0, 1.0])

        # Determine the scale factor from settings if not provided
        if scale_factor is None:
            scale_factor = self.settings.get("scale_factor", 1)

            # Apply rescaling normalization if scaling_type is unit
            rescaling_type = self.settings.get("rescaling_type", "none")
            scaling_type = self.settings.get("scaling_type", "unit")

            if rescaling_type != "none" and scaling_type == "unit":
                dist_category = "overall"
                if rescaling_type == "category_distance_based":
                    dist_category = "hard_process" if category == "MPI" else category

                # Use "non_beam_remnant" as a conservative default group
                ##furthest_dist = self.category_furthest_dist.get("non_beam_remnant", {}).get(
                ##    dist_category, 1.0  # safe fallback to avoid division by zero
                ##)
                furthest_dist = self.category_furthest_dist.get(remnant_key, {}).get(
                    dist_category, 1.0
                )

                scale_factor /= furthest_dist

        print(self.settings)
        displacement = direction  # * scale_factor
        print()
        print(f"direction: {direction}")
        print(f"scale_factor: {scale_factor}")
        print(f"Displacement before energy scaling: {displacement}")

        # Apply energy or log-energy scaling with auto-normalization
        rescaling_type = self.settings.get("rescaling_type", "none")
        dist_category = "overall"
        if rescaling_type == "category_distance_based":
            dist_category = "hard_process" if category == "MPI" else category

        if scaling_type == "energy":
            print("using energy scaling")
            displacement *= p.e()
        # norm = self.category_furthest_dist.get("non_beam_remnant", {}).get(dist_category, 1.0)
        # displacement /= norm

        elif scaling_type == "log_energy":
            print("using log energy scaling")
            displacement *= np.log(p.e())
        # norm = self.category_furthest_dist.get("non_beam_remnant", {}).get(dist_category, 1.0)
        # displacement /= norm

        elif scaling_type == "unit":
            print("using unit scaling (default)")

        print(scale_factor)
        # if rescale:
        # displacement = direction * scale_factor
        displacement *= self.settings.get("scale_factor", 1)

        # sys.exit()

        # Add base length if specified
        if self.settings.get("base_length", None) is not None:
            displacement += direction * self.settings.get("base_length", None)

        print()
        print(f"direction: {direction}")
        print(f"final displacement: {displacement}")

        end_position = start_position + displacement
        middle_position = start_position + displacement / 2.0

        return end_position, middle_position

    ###########################################################################
    ############################## Block 1 methods ############################
    ###########################################################################

    def get_initial_particles(self):
        """
        Identify initial outgoing particles based on their category and user
        settings.

        This method loops through all particles in the event and selects those
        that belong to one of the desired categories (hard process, beam
        remnants, MPI) and meet the corresponding settings criteria.

        Returns:
            list: A list of indices for the initial particles.
        """
        initial_particles = []
        # Iterate over all particles in the event
        for i in range(self.pythia.event.size()):
            prt = self.pythia.event[i]
            category = self.get_category(prt.status())
            # Check if the particle belongs to a desired category
            if category in ["hard_process", "beam_remnants", "MPI"]:
                # Only add particles if settings allow for beam remnants or MPI
                if (
                    (
                        category == "beam_remnants"
                        and self.settings.get("beam_remnant", False)
                    )
                    or (category == "MPI" and self.settings.get("mpi", False))
                    or (category == "hard_process")
                ):
                    initial_particles.append(i)
        return initial_particles

    def get_descendants(self, prt_idx, depth=0, effective_mother=None):
        """
        Recursively traverse the event tree starting from the given particle
        index.

        Returns:
            Tuple[List[Tuple[int, int]], Dict[int, List[int]]]:
                - A list of (prt_idx, depth) tuples for each processed
                  particle.
                - A mapping of particle indices to lists of effective mother
                  indices.
        """
        descendants = []  # List to hold (prt_idx, depth) tuples

        # Retrieve the particle and its properties.
        prt = self.pythia.event[prt_idx]
        status = prt.status()
        p = prt.p()
        pid = prt.id()
        energy = p.e()
        particle_name = self.pythia.particleData.name(pid)
        category = self.get_category(status, depth)

        # Apply boost if set, but skip for particles with status 63
        boost = self.compute_boost()
        if boost is not None and abs(status) != 63:
            p.bstback(boost)  # Apply boost correction to the momentum

        # For particles with status > 90, update effective mother and process
        # daughters, then return
        if abs(status) > 90:
            d1 = prt.daughter1()
            d2 = prt.daughter2()
            if d1 > 0:
                descendants += self._process_daughters(d1, d2, depth + 1, prt_idx)
            return descendants, self.particle_mother_mapping

        # Check for carbon copies if removal is enabled
        d1 = prt.daughter1()
        d2 = prt.daughter2()
        if self.settings.get("remove_copy", False):
            # Exclude status 23 and 63 particles from the carbon copy check
            if d1 > 0 and d2 == d1 and abs(status) not in [23, 63]:
                daughter = self.pythia.event[d1]
                # Check if the particle is a carbon copy (same PDG ID)
                if pid == daughter.id():
                    print(
                        f"{'  '*depth}Carbon Copy Detected: Skipping "
                        f"mother Index {prt_idx}"
                    )
                    # Continue processing with daughter; effective mother
                    # remains the same.
                    descendants += self._process_daughters(
                        d1, d1, depth, effective_mother
                    )
                    return descendants, self.particle_mother_mapping

        # Update the effective mother mapping if applicable.
        if effective_mother is not None and prt_idx != effective_mother:
            if prt_idx not in self.particle_mother_mapping:
                self.particle_mother_mapping[prt_idx] = []
            self.particle_mother_mapping[prt_idx].append(effective_mother)
            mother_energy = self.pythia.event[effective_mother].e()
        else:
            mother_energy = None

        tolerance = 1e-6  # GeV tolerance for numerical precision
        if mother_energy is not None and energy > mother_energy + tolerance:
            print(
                f"{'  ' * depth}Warning: Particle {prt_idx} has more "
                f"energy ({energy:.2e} GeV) than its mother "
                f"{effective_mother} ({mother_energy:.2e} GeV)"
            )

        # Record the current particle.
        descendants.append((prt_idx, depth))
        print(
            f"{'  ' * depth}{depth} Particle Index: {prt_idx}, "
            f"PDG ID: {pid}, Name: {particle_name}, Status: {status}, "
            f"Category: {category}"
        )
        print(
            f"{'  ' * depth}  Energy: {energy:.2e} GeV, px: {p.px():.2e}, "
            f"py: {p.py():.2e}, pz: {p.pz():.2e}"
        )

        # Handle gluons with shared daughters: only most energetic gluon proceeds
        if abs(pid) == 21 and (d1 > 0 or d2 > 0):
            daughters = []
            if d1 > 0:
                daughters.append(d1)
            if d2 > 0 and d2 != d1:
                daughters.append(d2)

            if not hasattr(self, "daughter_claims"):
                self.daughter_claims = {}

            can_proceed = True
            for d in daughters:
                existing = self.daughter_claims.get(d)
                if existing is not None:
                    existing_energy = self.pythia.event[existing].e()
                    if energy > existing_energy:
                        self.daughter_claims[d] = prt_idx
                    else:
                        can_proceed = False  # lower energy gluon, skip it
                else:
                    self.daughter_claims[d] = prt_idx

            if not can_proceed:
                print(
                    f"{'  '*depth}Skipping gluon {prt_idx}, daughter(s) already claimed by higher-energy gluon"
                )
                return descendants, self.particle_mother_mapping

        # Process daughter particles recursively.
        descendants += self._process_daughters(d1, d2, depth + 1, prt_idx)

        return descendants, self.particle_mother_mapping

    def assig_hadron_to_parton(self, boost=None):
        """
        Assign each hadron (with 80 < |status| < 90) to a parton mother based
        on the smallest delta R and positive cos(theta).

        Args:
            boost (pythia8.Vec4, optional): Boost vector to apply to particle
                momenta. Defaults to None.

        Returns:
            dict: The updated particle_mother_mapping, mapping hadron indices
            to a single best mother index.
        """

        def delta_r(p1, p2):
            """Calculate delta R using differences in eta and phi."""
            delta_eta = p1.eta() - p2.eta()
            delta_phi = np.abs(p1.phi() - p2.phi())
            if delta_phi > np.pi:
                delta_phi = 2 * np.pi - delta_phi
            return np.sqrt(delta_eta**2 + delta_phi**2)

        def cos_theta(p1, p2):
            """Calculate cos(theta) between two momentum vectors."""
            p1_vec = np.array([p1.px(), p1.py(), p1.pz()])
            p2_vec = np.array([p2.px(), p2.py(), p2.pz()])
            norm_p1 = np.linalg.norm(p1_vec)
            norm_p2 = np.linalg.norm(p2_vec)
            if norm_p1 > 0 and norm_p2 > 0:
                return np.dot(p1_vec, p2_vec) / (norm_p1 * norm_p2)
            else:
                return -1  # Handle cases with zero norm
                # (no meaningful direction)

        # Loop over each hadron in the particle-to-mother mapping.
        for hadron_index, mother_indices in self.particle_mother_mapping.items():
            hadron = self.pythia.event[hadron_index]
            hadron_p = hadron.p()
            if boost is not None:
                hadron_p.bstback(boost)
            status = hadron.status()

            # Process hadrons with 80 < id < 90 that have multiple mothers
            if 80 < abs(status) < 90 and len(mother_indices) > 1:
                print(
                    f"Processing hadron index {hadron_index} "
                    f"with status {status}. Mothers: {mother_indices}"
                )

                # Variables to track the best mother candidate
                best_mother_index = None
                smallest_delta_r = float("inf")
                best_cos_theta = -1  # cos(theta) must be positive to qualify

                # Fallback in case all mothers have non-positive cos(theta)
                fallback_mother_index = None
                fallback_smallest_delta_r = float("inf")

                # Evaluate each mother candidate
                for mother_index in mother_indices:
                    mother = self.pythia.event[mother_index]
                    mother_p = mother.p()
                    if boost is not None:
                        mother_p.bstback(boost)

                    # Compute delta R and cos(theta) between hadron and mother
                    dR = delta_r(hadron_p, mother_p)
                    cos_theta_value = cos_theta(hadron_p, mother_p)

                    print(
                        f"  Mother index {mother_index}, delta R: {dR:.4f}, "
                        f"cos(theta): {cos_theta_value:.4f}"
                    )

                    # Update fallback if this mother has a smaller delta R
                    if dR < fallback_smallest_delta_r:
                        fallback_smallest_delta_r = dR
                        fallback_mother_index = mother_index

                    # Only consider mothers with positive cos(theta)
                    if cos_theta_value > 0:
                        # Update the best candidate if delta R is lower or
                        # if equal check for a better cos(theta) value
                        if dR < smallest_delta_r or (
                            dR == smallest_delta_r and cos_theta_value > best_cos_theta
                        ):
                            smallest_delta_r = dR
                            best_mother_index = mother_index
                            best_cos_theta = cos_theta_value

                # Use fallback if no mother has a positive cos(theta)
                if best_mother_index is None and (fallback_mother_index is not None):
                    best_mother_index = fallback_mother_index
                    smallest_delta_r = fallback_smallest_delta_r
                    print(
                        f"  No positive cos(theta). Fallback to mother "
                        f"{best_mother_index} with smallest delta R of "
                        f"{smallest_delta_r:.4f}"
                    )

                # Update the mapping to keep only the best mother candidate.
                if best_mother_index is not None:
                    print(
                        f"  Keeping mother {best_mother_index} "
                        f"with delta R: {smallest_delta_r:.4f}, "
                        f"cos(theta): {best_cos_theta:.4f}"
                    )
                    self.particle_mother_mapping[hadron_index] = [best_mother_index]
                print("-" * 40)

        return self.particle_mother_mapping

    def get_color_neutral_object(self):
        """
        Connect partons into color-neutral objects using the
        mother-daughter mapping.

        Returns:
            tuple: A tuple containing:
                - color_neutral_objects (list): Groups (chains) of parton
                  indices that form color-neutral hadrons.
                - leftover_partons (list): Parton indices that could not be
                  grouped into a color-neutral object.
        """
        color_map = {}
        anti_color_map = {}
        partons_to_process = []  # List to store parton indices for later use

        # First, gather all relevant partons from the mother-daughter mapping
        parton_list = []

        print("particle_mother_mapping", self.particle_mother_mapping)

        # Combine keys and values from the mapping into one list
        all_particles = list(self.particle_mother_mapping.keys())
        for mother_indices in self.particle_mother_mapping.values():
            all_particles.extend(mother_indices)

        # Remove duplicates by converting to a set, then back to a list
        all_particles = list(set(all_particles))

        # Loop over the combined list to filter out partons with the
        # required statuses
        for prt_idx in all_particles:
            prt = self.pythia.event[prt_idx]
            status = abs(prt.status())

            # Include partons with status 71 or 74 directly
            if status in [71, 74]:
                parton_list.append(prt_idx)
                print("Added particle with status 71 or 74:", prt_idx)
            # For partons with status 63 or 23, check daughters
            elif status in [63, 23]:
                daughter1 = prt.daughter1()
                daughter2 = prt.daughter2()
                # Check if any daughter has status between 80 and 89
                if (
                    daughter1 > 0
                    and 80 <= abs(self.pythia.event[daughter1].status()) <= 89
                ):
                    parton_list.append(prt_idx)
                elif (
                    daughter2 > 0
                    and 80 <= abs(self.pythia.event[daughter2].status()) <= 89
                ):
                    parton_list.append(prt_idx)

        # Sort the list by particle index
        parton_list.sort()

        # Print sorted parton information for debugging
        for prt_idx in parton_list:
            prt = self.pythia.event[prt_idx]
            color_index = prt.col()
            anti_color_index = prt.acol()
            status = abs(prt.status())

            print(
                f"Particle {prt_idx}: Status = {status}, "
                f"Color = {color_index}, Anti-color = {anti_color_index}"
            )

            partons_to_process.append(prt_idx)  # Track for processing

            # Build the color_map.
            if color_index != 0:
                if color_index not in color_map:
                    color_map[color_index] = []
                color_map[color_index].append(prt_idx)

            # Build the anti_color_map.
            if anti_color_index != 0:
                if anti_color_index not in anti_color_map:
                    anti_color_map[anti_color_index] = []
                anti_color_map[anti_color_index].append(prt_idx)

        # Copy the list of partons to keep track of leftover partons
        leftover_partons = partons_to_process.copy()

        # Process the partons to form color-neutral chains
        while partons_to_process:
            chain = []
            current_parton = partons_to_process.pop(0)
            chain.append(current_parton)

            # Get the color and anti-color indices of the current parton
            color_index = self.pythia.event[current_parton].col()
            anti_color_index = self.pythia.event[current_parton].acol()

            # Connect partons forward based on color matching
            while color_index in anti_color_map and anti_color_map[color_index]:
                next_parton = anti_color_map[color_index].pop(0)
                chain.append(next_parton)
                partons_to_process.remove(next_parton)
                color_index = self.pythia.event[next_parton].col()

            # Connect partons backward based on anti-color matching
            while anti_color_index in color_map and color_map[anti_color_index]:
                prev_parton = color_map[anti_color_index].pop(0)
                chain.insert(0, prev_parton)
                partons_to_process.remove(prev_parton)
                anti_color_index = self.pythia.event[prev_parton].acol()

            # Append the formed chain to the list of color-neutral objects
            self.color_neutral_objects.append(chain)

            # Remove the partons that have been grouped from leftovers
            for parton in chain:
                if parton in leftover_partons:
                    leftover_partons.remove(parton)

        return self.color_neutral_objects, leftover_partons

    def build_particle_tree(self, initial_particles=[]):
        """
        Process all initial particles to compute their descendants and update
        the mother mapping.

        For each particle in initial_particles, this method retrieves its
        descendants (with depth) via get_descendants and updates the
        particle_mother_mapping accordingly.

        Args:
            initial_particles (iterable): A list (or iterable) of particle
                indices that serve as the starting points for tree building.

        Returns:
            tuple: A tuple containing:
                - list_descendants_with_depth: A list of lists, each holding
                  (prt_idx, depth) tuples.
                - particle_mother_mapping: The updated mapping of particles to
                  their effective mothers.
        """
        if not initial_particles:
            initial_particles = self.get_initial_particles()

        for idx in initial_particles:
            descendants_with_depth, updated_mapping = self.get_descendants(
                idx, depth=0, effective_mother=idx
            )
            self.list_descendants_with_depth.append(descendants_with_depth)
            self.particle_mother_mapping.update(updated_mapping)
        return self.list_descendants_with_depth, self.particle_mother_mapping

    ###########################################################################
    ############################## Block 2 methods ############################
    ###########################################################################

    def compute_positions(
        self, initial_particles, rescale=False, x_shift=None, y_shift=None, z_shift=None
    ):
        """
        Compute the start, middle, and end positions of particles based on
        their momentum.

        Depending on the value of rescale, the computed positions are stored in
        either:
          - self.rescaled_positions (if rescale is True)
          - self.particle_positions (if rescale is False)

        Args:
            initial_particles (iterable): List or iterable of particle indices to
                process first.
            rescale (bool, optional): Whether to apply rescaling based on the
                furthest distance and adjust the start position for MPI particles.
                Default is False.
            x_shift (float, optional): Shift in the x-direction for MPI particles
                when rescaling.
            y_shift (float, optional): Shift in the y-direction for MPI particles
                when rescaling.
            z_shift (float, optional): Shift in the z-direction for MPI particles
                when rescaling.

        Returns:
            dict: A dictionary mapping each particle index to another
            dictionary with keys 'start', 'middle', and 'end' representing
            the computed positions.
        """
        # Choose the correct dictionary based on rescaling
        positions = self.rescaled_positions if rescale else self.particle_positions

        # Process initial particles first
        for prt_idx in initial_particles:
            prt = self.pythia.event[prt_idx]
            p = prt.p()
            # Apply boost transformation if available
            boost = self.compute_boost()
            if boost is not None:
                p.bstback(boost)

            # Set default start position at the origin
            start_position = np.array([0.0, 0.0, 0.0])
            # Use absolute status for rescaled positions to ensure proper
            # category determination
            status = prt.status() if not rescale else abs(prt.status())
            category = self.get_category(status)

            # Initialize scale factor from settings
            scale_factor = self.settings.get("scale_factor", 1)
            if rescale:
                #    print('I am rescaling')
                #    sys.exit()
                print("self.futhest_dist", self.category_furthest_dist)
                if self.settings.get("rescaling_type") == "total_distance_based":
                    scaling_type = "overall"
                elif self.settings.get("rescaling_type") == "category_distance_based":
                    scaling_type = category
                    if category == "MPI":
                        scaling_type = "hard_process"

                # Determine if the particle is part of a beam remnant
                is_beam_remnant = self.has_beam_remnant_ancestor(prt_idx)
                # Get the furthest distance based on the particle category
                key = "beam_remnant" if is_beam_remnant else "non_beam_remnant"

                furthest_dist = self.category_furthest_dist[key].get(
                    category, self.category_furthest_dist[key][scaling_type]
                )
                # Compute the scale factor
                scale_factor = self.settings.get("scale_factor", 1) / furthest_dist

                # For MPI particles, adjust the start position based on the x
                # component of momentum
                if category == "MPI":
                    if p.px() >= 0:
                        start_position = np.array([x_shift, y_shift, z_shift])
                    else:
                        start_position = np.array([-x_shift, -y_shift, z_shift])

            # Estimate the end and middle positions using momentum
            is_beam_remnant = self.has_beam_remnant_ancestor(prt_idx)
            remnant_key = "beam_remnant" if is_beam_remnant else "non_beam_remnant"
            end_position, middle_position = self.compute_mid_end_pos(
                p,
                start_position,
                category=category,
                scale_factor=None,
                rescale=rescale,
                remnant_key=remnant_key,
            )

            # self.compute_mid_end_pos(
            #     p, start_position, category=category, scale_factor=scale_factor
            # )
            #  print('end_position', end_position)
            #  print('middle_position', middle_position)
            #  sys.exit()
            # Store computed positions for the particle
            positions[prt_idx] = {
                "start": start_position,
                "middle": middle_position,
                "end": end_position,
            }

        # ------------------------------
        # Process remaining particles using the mother mapping
        # ------------------------------
        sorted_particles = dict(sorted(self.particle_mother_mapping.items()))
        for prt_idx, mother_indices in sorted_particles.items():
            prt = self.pythia.event[prt_idx]
            p = prt.p()
            # Apply boost transformation if available
            boost = self.compute_boost()
            if boost is not None:
                p.bstback(boost)

            # Determine the particle's status and category
            status = prt.status() if not rescale else abs(prt.status())
            category = self.get_category(status)

            # Initialize scale factor.
            scale_factor = self.settings.get("scale_factor", 1)
            if rescale:
                # Determine beam remnant status and
                # retrieve the furthest distance
                is_beam_remnant = self.has_beam_remnant_ancestor(prt_idx)
                key = "beam_remnant" if is_beam_remnant else "non_beam_remnant"

                furthest_dist = self.category_furthest_dist[key].get(
                    category, self.category_furthest_dist[key]["overall"]
                )
                # Recompute the scale factor based on the furthest distance
                scale_factor = self.settings.get("scale_factor", 1) / furthest_dist

            # Determine the start position from the mother's end position
            if mother_indices:
                mother_index = mother_indices[0]
                # Ensure that the mother's position has been computed
                if mother_index not in positions:
                    raise ValueError(
                        f"Mother index {mother_index} " "has not been processed yet."
                    )
                start_position = positions[mother_index]["end"]
            else:
                # If no mother is present, default to the origin
                start_position = np.array([0.0, 0.0, 0.0])

            # Estimate the end and middle positions using momentum
            is_beam_remnant = self.has_beam_remnant_ancestor(prt_idx)
            remnant_key = "beam_remnant" if is_beam_remnant else "non_beam_remnant"
            end_position, middle_position = self.compute_mid_end_pos(
                p,
                start_position,
                category=category,
                scale_factor=None,
                remnant_key=remnant_key,
            )

            ##self.compute_mid_end_pos(
            ##     p, start_position,
            ##     category=category,
            ##     scale_factor=scale_factor
            ##)
            # Store computed positions for the particle.
            positions[prt_idx] = {
                "start": start_position,
                "middle": middle_position,
                "end": end_position,
            }

        # Optional debug printing when rescaling is enabled
        if rescale:
            print("Initial particles processed.")
            for particle in self.particle_positions:
                print(
                    f"Particle index: {particle}, Positions: "
                    f"{self.particle_positions[particle]}"
                )

        return positions

    def find_furthest_dist_from_positions(self):
        """
        Compute the furthest and shortest distances for particle positions and
        separate them based on beam remnant association.

        Returns:
            dict: A dictionary containing, for each of 'beam_remnant' and
            'non_beam_remnant', the distance (furthest minus shortest)
            for each
            category (hard_process, parton_shower, hadronization, overall).
        """
        categories_non_beam = [
            "hard_process",
            "parton_shower",
            "hadronization",
            "overall",
        ]
        categories_beam = ["beam_remnants", "parton_shower", "hadronization", "overall"]

        distnaces = {
            "non_beam_remnant": {
                category: {"furthest": 0.0, "shortest": float("inf")}
                for category in categories_non_beam
            },
            "beam_remnant": {
                category: {"furthest": 0.0, "shortest": float("inf")}
                for category in categories_beam
            },
        }

        def update_distances(prt_idx, distance, category_key, is_beam_remnant):
            """
            Update the furthest and shortest distances for a given category.

            Args:
                prt_idx (int): The index of the particle.
                distance (float): The computed distance from the origin.
                category_key (str): The particle category.
                is_beam_remnant (bool): True if the particle (or its ancestor)
                    is a beam remnant.
            """
            remnant_key = "beam_remnant" if is_beam_remnant else "non_beam_remnant"
            if category_key == "MPI":
                category_key = "parton_shower"
            category_data = distnaces[remnant_key][category_key]

            # Update furthest distance if the new distance is larger
            if distance > category_data["furthest"]:
                category_data["furthest"] = distance

            # Update shortest distance if the new distance is smaller
            if distance < category_data["shortest"]:
                category_data["shortest"] = distance

        # Loop through each descendant list with depth
        for descendants_with_depth in self.list_descendants_with_depth:
            # Loop through each (prt_idx, depth) tuple
            for prt_idx, depth in descendants_with_depth:
                # Check if the particle's position has been computed
                if prt_idx in self.particle_positions:
                    # Retrieve start and end positions
                    start_position = self.particle_positions[prt_idx]["start"]
                    end_position = self.particle_positions[prt_idx]["end"]

                    # Compute distances from the origin
                    start_distance = np.linalg.norm(start_position)
                    end_distance = np.linalg.norm(end_position)

                    # Retrieve the particle from the event
                    particle = self.pythia.event[prt_idx]
                    status = abs(particle.status())

                    # Determine if the particle or
                    # its ancestors are beam remnants
                    is_beam_remnant = self.has_beam_remnant_ancestor(prt_idx)

                    # Get the category of the particle
                    category_key = self.get_category(status)
                    if category_key:
                        # Update distances for the given category
                        update_distances(
                            prt_idx, end_distance, category_key, is_beam_remnant
                        )
                        update_distances(
                            prt_idx, start_distance, category_key, is_beam_remnant
                        )

                    # Always update for the overall distance category
                    update_distances(prt_idx, end_distance, "overall", is_beam_remnant)
                    update_distances(
                        prt_idx, start_distance, "overall", is_beam_remnant
                    )

        # Calculate the difference between furthest and
        # shortest for each category
        for remnant_key in ["non_beam_remnant", "beam_remnant"]:
            categories = (
                categories_non_beam
                if remnant_key == "non_beam_remnant"
                else categories_beam
            )
            self.category_furthest_dist[remnant_key] = {}

            for category in categories:
                category_data = distnaces[remnant_key][category]
                furthest_minus_shortest = (
                    category_data["furthest"] - category_data["shortest"]
                )

                # Store the difference for this category
                self.category_furthest_dist[remnant_key][
                    category
                ] = furthest_minus_shortest

        return self.category_furthest_dist

    def get_color_connection_info(self, bow_height=8):
        """
        Connect the end points of particles within color-neutral objects and
        compute bow-shaped middle points for visualization.

        Args:
            bow_height (float, optional): The offset height used to create the
                bow-shaped curvature. Default is 8.

        Returns:
            list: A list of dictionaries, each containing track connection info
            with keys: 'category', 'from', 'to', 'neutral object', 'color', and
            'pos' (positions list).
        """
        # Determine the number of color-neutral objects
        num_objects = len(self.color_neutral_objects)

        # Generate unique colors for each object
        colors = self.generate_unique_colors(num_objects)

        # Loop over each color-neutral object
        for k, color_neutral_object in enumerate(self.color_neutral_objects):
            # color = colors[k]  # Assign a unique color for this group
            # if self.highlight_category is not None and all(cat != self.highlight_category for cat in group_categories):
            if (
                self.highlight_category is not None
                and "color_connection" != self.highlight_category
            ):
                color = "D3D3D3"  # greyed out if no parton matches the highlight
            else:
                color = colors[k]  # Assign a unique color for this group

            color_neutral_details = []
            # Collect details for each parton in the object
            for parton in color_neutral_object:
                prt = self.pythia.event[parton]
                detail = (
                    f"idx: {parton}, id: "
                    f"{self.pythia.particleData.name(prt.id())} "
                    f"({prt.id()}), "
                    f"color: {prt.col()}, "
                    f"acol: {prt.acol()};  "
                )
                color_neutral_details.append(detail)

            # Connect consecutive partons in the color-neutral object
            for i in range(len(color_neutral_object) - 1):
                parton_a = color_neutral_object[i]
                parton_b = color_neutral_object[i + 1]

                # Get the end positions for both partons
                start_position = self.rescaled_positions[parton_a]["end"]
                end_position = self.rescaled_positions[parton_b]["end"]

                # Calculate the basic middle position
                basic_middle_position = start_position + end_position / 2.0

                # Calculate a perpendicular vector for the bow offset
                direction_vector = end_position - start_position
                norm = np.linalg.norm(direction_vector)
                if norm > 0:
                    direction_vector = direction_vector / norm  # Normalize
                    # Compute a perpendicular vector
                    perpendicular_vector = np.cross(direction_vector, [0, 0, 1])
                    # Adjust the middle position to create a bow shape
                    bow_middle_position = (
                        basic_middle_position + perpendicular_vector * bow_height
                    )
                else:
                    # Fallback if the direction vector norm is zero
                    bow_middle_position = basic_middle_position

                ########################

                # Retrieve particle information for both partons
                prt_a = self.pythia.event[parton_a]
                prt_b = self.pythia.event[parton_b]

                from_info = (
                    f"idx: {parton_a}, id: "
                    f"{self.pythia.particleData.name(prt_a.id())} "
                    f"({prt_a.id()}), "
                    f"color: {prt_a.col()}, "
                    f"acol: {prt_a.acol()}"
                )
                to_info = (
                    f"idx: {parton_b}, id: "
                    f"{self.pythia.particleData.name(prt_b.id())} "
                    f"({prt_b.id()}), "
                    f"color: {prt_b.col()}, "
                    f"acol: {prt_b.acol()}"
                )

                ########################

                # Build the connection information.
                connection_info = {
                    "category": "color connection",
                    "from": from_info,
                    "to": to_info,
                    "neutral object": "\n" + "\n".join(color_neutral_details),
                    "color": color,
                    "pos": [
                        start_position.tolist(),
                        bow_middle_position.tolist(),
                        end_position.tolist(),
                    ],
                }

                # Append the connection info to the list
                self.track_color_connection_info.append(connection_info)

        return self.track_color_connection_info

    def process_particle(self):
        """
        Iterate through precomputed particle positions and store particle
        information (momentum, category, energy, etc.) in track_info format.
        """
        particle_info_list = []

        def get_status_sentence(status):
            abs_status = abs(status)  # Use absolute value for status
            return status_meaning.get(str(abs_status), "Unknown status code")

        vertices_list = []
        #    missing_energy_list = []

        # Iterate over precomputed positions
        for prt_idx, positions in self.rescaled_positions.items():
            # Get the particle from the Pythia event
            prt = self.pythia.event[prt_idx]
            p = prt.p()

            # Apply boost if provided
            boost = self.compute_boost()
            if boost is not None:
                p.bstback(boost)

            # Retrieve particle properties
            pid = prt.id()
            particle_name = self.pythia.particleData.name(pid)
            status = prt.status()
            status_sentence = get_status_sentence(status)
            category = self.get_category(status)
            px = "{:.2e}".format(p.px())
            py = "{:.2e}".format(p.py())
            pz = "{:.2e}".format(p.pz())
            e = "{:.2e}".format(p.e())
            eta = "{:.2e}".format(p.eta())
            phi = "{:.2e}".format(p.phi())
            m = "{:.2e}".format(prt.m())
            col = prt.col()
            acol = prt.acol()
            mom1 = prt.mother1()
            mom2 = prt.mother2()
            daughter1 = prt.daughter1()
            daughter2 = prt.daughter2()
            xProd = "{:.2e}".format(prt.xProd())
            yProd = "{:.2e}".format(prt.yProd())
            zProd = "{:.2e}".format(prt.zProd())
            tProd = "{:.2e}".format(prt.tProd())
            tau = "{:.2e}".format(prt.tau())
            pol = "{:.2e}".format(prt.pol())
            scale = "{:.2e}".format(prt.scale())
            charge = prt.charge()
            isFinal = prt.isFinal()
            # isDecayed = prt.isDecayed()
            m2 = "{:.2e}".format(prt.m2())
            pT = "{:.2e}".format(p.pT())
            pT2 = "{:.2e}".format(prt.pT2())
            mT = "{:.2e}".format(prt.mT())
            mT2 = "{:.2e}".format(prt.mT2())
            pAbs = "{:.2e}".format(prt.pAbs())
            pAbs2 = "{:.2e}".format(prt.pAbs2())
            theta = "{:.2e}".format(p.theta())
            thetaXZ = "{:.2e}".format(prt.thetaXZ())
            eCalc = "{:.2e}".format(prt.eCalc())

            # Assign a color based on the particle category
            color = self.assign_color(category)

            # If the particle is a gluon (PDG id 21 or -21) and
            # one of its daughter indices is 0, lighten the color.
            if abs(pid) == 21 and (daughter1 == 0 or daughter2 == 0):
                color = self.lighten_color(color, factor=0.5)
                # update category name
                category += " , reabsorbed gluon"

            # Retrieve precomputed positions
            start_position = positions["start"]
            middle_position = positions["middle"]
            end_position = positions["end"]

            ######### Lepton case #########

            # Compute the furthest distance from the center for any particle
            furthest_distance = max(
                np.linalg.norm(pos["end"])  # Distance from (0,0,0) to end position
                for pos in self.rescaled_positions.values()
            )

            # Check if the particle is a lepton and lighten its color
            if abs(pid) in [11, 13, 15, 12, 14, 16]:
                color = self.lighten_color(color, factor=0.45)  # Lighten color by 70%

            # Rescale lepton tracks so they extend to the furthest particle distance
            if abs(pid) in [11, 13, 15, 12, 14, 16]:
                start_to_end_vector = end_position - start_position
                start_distance = np.linalg.norm(start_position)

                if np.linalg.norm(start_to_end_vector) > 0:
                    scale_factor = (
                        furthest_distance - start_distance
                    ) / np.linalg.norm(start_to_end_vector)
                    end_position = start_position + start_to_end_vector * scale_factor
                    middle_position = (
                        start_position + (end_position - start_position) / 2
                    )

            # Make non-highlighted categories grey
            if (
                self.highlight_category is not None
                and category != self.highlight_category
            ):
                color = self.lighten_color("808080", factor=0.3)

            if abs(pid) in [11, 13, 15, 12, 14, 16]:
                if (
                    self.highlight_category is None
                    or category != self.highlight_category
                ):
                    color = self.lighten_color("808080", factor=0.3)

            # Store the particle information in track_info format
            track_info = {
                "category": category,
                "label": str(prt_idx),
                "status": str(status) + ": " + status_sentence,
                "id": f"{particle_name} ({pid})",
                "px": px,
                "py": py,
                "pz": pz,
                "e": e,
                "eta": eta,
                "phi": phi,
                "m": m,
                "col": col,
                "acol": acol,
                "mom1": mom1,
                "mom2": mom2,
                "daughter1": daughter1,
                "daughter2": daughter2,
                "xProd": xProd,
                "yProd": yProd,
                "zProd": zProd,
                "tProd": tProd,
                "tau": tau,
                "pol": pol,
                "scale": scale,
                "charge": charge,
                "isFinal": isFinal,
                "m2": m2,
                "pT": pT,
                "pT2": pT2,
                "mT": mT,
                "mT2": mT2,
                "pAbs": pAbs,
                "pAbs2": pAbs2,
                "theta": theta,
                "thetaXZ": thetaXZ,
                "eCalc": eCalc,
                "color": color,
                "pos": [
                    start_position.tolist(),
                    middle_position.tolist(),
                    end_position.tolist(),
                ],
            }

            # Create vertex info
            vertices_info = {"pos": start_position.tolist()}

            # Append to the JSON data and list
            vertices_list.append(vertices_info)

            if category not in self.json_data["Test, tracks playground"]["Tracks"]:
                self.json_data["Test, tracks playground"]["Tracks"][category] = []

            self.json_data["Test, tracks playground"]["Tracks"][category].append(
                track_info
            )
            particle_info_list.append(track_info)

        self.json_data["Test, tracks playground"]["Vertices"]["TestVertices"].extend(
            vertices_list
        )

        return self.json_data, particle_info_list, vertices_list

    def merge_color_connections_into_tracks(self):
        """
        Merge the color connection information into the 'Tracks' section of the
        JSON data after 'TestTracks'.

        This method ensures that the 'Tracks' section exists and then appends
        each connection from self.track_color_connection_info into the
        'TestTracks' list of the JSON data.

        Returns:
            dict: The updated JSON data structure with
                  merged color connections.
        """
        # Ensure that the 'Tracks' section and '
        # TestTracks' exists in the JSON data
        if "Tracks" not in self.json_data["Test, tracks playground"]:
            self.json_data["Test, tracks playground"]["Tracks"] = {}

        if "TestTracks" not in self.json_data["Test, tracks playground"]["Tracks"]:
            self.json_data["Test, tracks playground"]["Tracks"]["TestTracks"] = []

        if (
            "Collor Connection"
            not in self.json_data["Test, tracks playground"]["Tracks"]
        ):
            self.json_data["Test, tracks playground"]["Tracks"][
                "Collor Connection"
            ] = []
        # Loop through each connection and add it to TestTracks
        for connection in self.track_color_connection_info:
            self.json_data["Test, tracks playground"]["Tracks"][
                "Collor Connection"
            ].append(connection)

        return self.json_data

    def get_json(self):
        """
        Main method for processing a single event. Coordinates the workflow
        across different processing blocks.
        """
        ######### Block 1: Build the effective particle tree ##############

        # Get initial particles.
        initial_particles = self.get_initial_particles()

        # Build the particle tree.
        (
            self.list_descendants_with_depth,
            self.particle_mother_mapping,
        ) = self.build_particle_tree(initial_particles)

        # Assign hadron daughter to parton mother based on angle.
        self.particle_mother_mapping = self.assig_hadron_to_parton()

        # Get color connections if enabled.
        if self.settings["color_connection"]:
            (
                self.color_neutral_objects,
                leftover_partons,
            ) = self.get_color_neutral_object()

        ########### Block 2: Get and assign particle information #############
        # Shifts for MPI (to be accessed directly in methods later)
        x_shift = self.settings["mpi_location"][0]
        y_shift = self.settings["mpi_location"][1]
        z_shift = self.settings["mpi_location"][2]

        # Get particle positions (without rescaling).
        self.particle_positions = self.compute_positions(
            initial_particles, rescale=False
        )

        print("Particle Positions (no rescaling):")
        for idx, pos in self.particle_positions.items():
            print(f"Particle {idx}:")
            print(f"  Start : {pos['start']}")
            print(f"  Middle: {pos['middle']}")
            print(f"  End   : {pos['end']}")

        # sys.exit()

        # Find furthest distances for different categories.
        self.category_furthest_dist = self.find_furthest_dist_from_positions()

        # Rescale positions with provided shifts.
        self.rescaled_positions = self.compute_positions(
            initial_particles,
            rescale=True,
            x_shift=x_shift,
            y_shift=y_shift,
            z_shift=z_shift,
        )

        print("Particle Positions (rescaling):")
        for idx, pos in self.rescaled_positions.items():
            print(f"Particle {idx}:")
            print(f"  Start : {pos['start']}")
            print(f"  Middle: {pos['middle']}")
            print(f"  End   : {pos['end']}")
        # print('I am here')
        # sys.exit()

        # Get color connection info if enabled.
        if self.settings["color_connection"]:
            self.track_color_connection_info = self.get_color_connection_info()

        # Get particle information and store it.
        self.json_data, _, _ = self.process_particle()
        self.json_data = self.merge_color_connections_into_tracks()

        # Ensure 'files/' directory exists
        output_dir = "files"
        os.makedirs(output_dir, exist_ok=True)

        # Construct full path to save the file in 'files/' folder
        file_path = os.path.join(output_dir, self.file_name)

        # Save the JSON data
        with open(file_path, "w") as f:
            json.dump(self.json_data, f, indent=4)

        # # Save the JSON data.
        # with open(self.file_name, "w") as f:
        #     json.dump(self.json_data, f, indent=4)

        return self.json_data


def str2bool(v):
    """
    Convert a string to a boolean.

    Accepts: 'yes', 'true', 't', 'y', '1' for True and
             'no', 'false', 'f', 'n', '0' for False.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ("yes", "true", "t", "y", "1"):
        return True
    elif v_lower in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    """
    Parse command-line arguments for configuring Pythia and visualization
    settings.
    """
    parser = argparse.ArgumentParser(description="Run Pythia visualization tool.")

    # Both arguments are now optional but at least one must be provided.
    parser.add_argument(
        "--read_strings",
        nargs="+",
        type=str,
        help="List of Pythia readString commands to configure Pythia.",
    )
    parser.add_argument(
        "--command_file",
        nargs="+",
        type=str,
        help="Paths to one or more Pythia .cmnd files containing commands.",
    )

    # Visualization settings, using str2bool for booleans.
    parser.add_argument(
        "--beam_remnant",
        type=str2bool,
        default=True,
        help="Add beam remnant (default: True).",
    )
    parser.add_argument(
        "--remove_copy",
        type=str2bool,
        default=True,
        help="Remove carbon copies (default: True).",
    )
    parser.add_argument(
        "--color_connection",
        type=str2bool,
        default=True,
        help="Enable color connection (default: True).",
    )
    parser.add_argument(
        "--mpi",
        type=str2bool,
        default=True,
        help="Include multi-parton interactions (default: True).",
    )
    parser.add_argument(
        "--scale_factor", type=float, default=1, help="Scale factor for visualization."
    )
    parser.add_argument(
        "--boost_mode",
        type=str,
        default="cm_incoming",
        choices=["None", "cm_incoming", "cm_outgoing"],
        help="Boost mode: None, cm_incoming, or cm_outgoing.",
    )
    parser.add_argument(
        "--scaling_type",
        type=str,
        default="unit",
        choices=["unit", "energy", "log_energy"],
        help="Scaling type: unit, energy, or log_energy.",
    )
    parser.add_argument(
        "--rescaling_type",
        type=str,
        default="category_distance_based",
        choices=["none", "total_distance_based", "category_distance_based"],
        help="Rescaling type.",
    )
    parser.add_argument(
        "--base_length", type=float, default=40, help="Base length added to each track."
    )
    # MPI location settings: x, y, and z coordinates.
    parser.add_argument(
        "--mpi_x",
        type=float,
        default=19.2,
        help="MPI location x coordinate (default: 19.2).",
    )
    parser.add_argument(
        "--mpi_y",
        type=float,
        default=8.9,
        help="MPI location y coordinate (default: 8.9).",
    )
    parser.add_argument(
        "--mpi_z",
        type=float,
        default=0.0,
        help="MPI location z coordinate (default: 0.0).",
    )

    # Generate a timestamped default file name.
    timestamp = datetime.datetime.now().strftime("date_%Y_%m_%d_time_%H:%M:%S")
    default_file_name = f"visualization_{timestamp}.json"

    parser.add_argument(
        "--file_name",
        type=str,
        default=default_file_name,
        help=f"Output file name (default: {default_file_name}).",
    )

    args = parser.parse_args()

    # Ensure that at least one source of commands is provided.
    if not (args.command_file or args.read_strings):
        parser.error(
            "At least one of --command_file or" " --read_strings must be provided."
        )

    return args


def load_commands_from_file(file_path):
    """
    Load Pythia commands from a .cmnd file.

    The file should contain one command per line. Blank lines and lines
    not beginning with a letter or digit are ignored. Inline comments
    (anything after an exclamation point) are removed.
    """
    commands = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Process only lines that start with a letter or digit.
            if line and line[0].isalnum():
                # Remove inline comments by splitting at the first '!'
                line = line.split("!", 1)[0].strip()
                if line:  # Ensure the line is not empty
                    # after removing the comment.
                    commands.append(line)
    return commands


def configure_pythia(pythia, commands):
    """
    Configure Pythia using a list of commands.
    """
    for command in commands:
        print(f"Executing: {command}")
        pythia.readString(command)


def open_website():
    url = "https://hepsoftwarefoundation.org/phoenix/playground"
    try:
        if not webbrowser.open(url, new=2):
            print("Failed to open URL. Please check your browser settings.")
    except Exception as e:
        print(f"Error opening website: {e}")


# Main execution code
if __name__ == "__main__":
    args = parse_arguments()

    # Combine commands from files and read_strings.
    commands = []

    if args.command_file:
        for file_path in args.command_file:
            if not os.path.exists(file_path):
                sys.exit(f"Error: Command file {file_path} does not exist.")
            file_commands = load_commands_from_file(file_path)
            commands.extend(file_commands)

    if args.read_strings:
        commands.extend(args.read_strings)

    # Debug: print the final list of commands.
    print("Final list of Pythia commands:")
    for cmd in commands:
        print(f"  {cmd}")

    if args.scaling_type == "log_energy":
        args.scale_factor /= 5
    elif args.scaling_type == "energy":
        args.scale_factor /= 50

    # Prepare settings for visualization.
    settings = {
        "remove_copy": args.remove_copy,  # Removes carbon copies
        "beam_remnant": args.beam_remnant,  # Includes beam remnant
        "scale_factor": args.scale_factor,  # Scale factor for visualization
        "boost_mode": args.boost_mode,  # Boost mode: None, "cm_incoming",
        # or "cm_outgoing"
        "scaling_type": args.scaling_type,  # Scaling type: "unit", "energy",
        # or "log_energy"
        "rescaling_type": args.rescaling_type,  # Rescaling: "none",
        # "total_distance_based",
        #  "category_distance_based"
        "base_length": args.base_length,  # Base length added to each track
        "color_connection": args.color_connection,  # Include color connection
        "mpi": args.mpi,  # Include multi-parton interaction
        "mpi_location": [args.mpi_x, args.mpi_y, args.mpi_z],  # MPI starting
        # location
    }

    print("Visualization settings:")
    print(settings)

    # Create and configure a Pythia instance.
    pythia = pythia8.Pythia()
    configure_pythia(pythia, commands)
    pythia.init()

    # Process an event (you can adjust the number of events as needed).
    pythia.next()
    #    pythia.next()
    pythia.event.list()

    #   sys.exit()

    file_name = args.file_name

    # Initialize the Visualization class
    visualization = Visualization(pythia, settings, file_name)
    result = visualization.get_json()

    open_website()
