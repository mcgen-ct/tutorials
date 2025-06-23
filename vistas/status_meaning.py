# status_meaning.py - Dictionary of Pythia event status codes and descriptions.
# Copyright (C) 2024 Torbjorn Sjostrand.
# Authors: Phil Ilten, Ahmed Youssef, Jure Zupan
# This software is licensed under the GNU GPL v2 or later. See COPYING for
# details. Please respect the MCnet Guidelines, see GUIDELINES for details.

# Keywords: Pythia8; particle status; event categorization; Monte Carlo sim;

# This module contains a dictionary (`STATUS_MEANINGS`) that maps Pythia event
# status codes to human-readable descriptions. It replaces the need for an
# external JSON file and allows easy access to event status information.

status_meaning = {
    "11": "beam particles; the event as a whole",
    "12": "beam particles; incoming beam",
    "13": "beam particles; incoming beam-inside-beam (e.g. gamma inside e)",
    "14": "beam particles; outgoing elastically scattered",
    "15": "beam particles; outgoing diffractively scattered",
    "21": "particles of the hardest subprocess; incoming particle of the hardest subprocess",
    "22": "particles of the hardest subprocess; intermediate (intended to have preserved mass)",
    "23": "particles of the hardest subprocess; outgoing particle of the hardest subprocess",
    "24": "particles of the hardest subprocess; outgoing, nonperturbatively kicked out in diffraction",
    "31": "particles of subsequent subprocesses; incoming particle of subsequent subprocess",
    "32": "particles of subsequent subprocesses; intermediate (intended to have preserved mass)",
    "33": "particles of subsequent subprocesses; outgoing particle of subsequent subprocess",
    "34": "particles of subsequent subprocesses; incoming that has already scattered",
    "41": "particles produced by initial-state-showers; incoming on spacelike main branch",
    "42": "particles produced by initial-state-showers; incoming copy of recoiler",
    "43": "particles produced by initial-state-showers; outgoing produced by a branching",
    "44": "particles produced by initial-state-showers; outgoing shifted by a branching",
    "45": "particles produced by initial-state-showers; incoming rescattered parton, with changed kinematics owing to ISR in the mother system",
    "46": "particles produced by initial-state-showers; incoming copy of recoiler when this is a rescattered parton",
    "47": "particles produced by initial-state-showers; a W or Z gauge boson produced in the shower evolution",
    "49": "a special state in the evolution, where E^2 - p^2 = m^2 is not fulfilled",
    "51": "particles produced by final-state-showers; outgoing produced by parton branching",
    "52": "particles produced by final-state-showers; outgoing copy of recoiler, with changed momentum",
    "53": "particles produced by final-state-showers; copy of recoiler when this is incoming parton, with changed momentum",
    "54": "particles produced by final-state-showers; copy of a recoiler, when in the initial state of a different system from the radiator",
    "55": "particles produced by final-state-showers; copy of a recoiler, when in the final state of a different system from the radiator",
    "56": "particles produced by final-state-showers; a W or Z gauge boson produced in a shower branching (special case of 51)",
    "59": "particles produced by final-state-showers; a special state in the evolution, where E^2 - p^2 = m^2 is not fulfilled",
    "61": "particles produced by beam-remnant treatment; incoming subprocess particle with primordial kT included",
    "62": "particles produced by beam-remnant treatment; outgoing subprocess particle with primordial kT included",
    "63": "particles produced by beam-remnant treatment; outgoing beam remnant",
    "64": "particles produced by beam-remnant treatment; copied particle with new colour according to the colour configuration of the beam remnant",
    "71": "partons in preparation of hadronization process; copied partons to collect into contiguous colour singlet",
    "72": "partons in preparation of hadronization process; copied recoiling singlet when ministring collapses to one hadron and momentum has to be reshuffled",
    "73": "partons in preparation of hadronization process; combination of very nearby partons into one",
    "74": "partons in preparation of hadronization process; combination of two junction quarks (+ nearby gluons) to a diquark",
    "75": "partons in preparation of hadronization process; gluons split to decouple a junction-antijunction pair",
    "76": "partons in preparation of hadronization process; partons with momentum shuffled or a new colour to decouple junction-antijunction structures",
    "77": "partons in preparation of hadronization process; temporary opposing parton when fragmenting first two strings in to junction",
    "78": "partons in preparation of hadronization process; temporary combined diquark end when fragmenting last string in to junction",
    "79": "partons in preparation of hadronization process; copy of particle with new colour indices after the colour reconnection",
    "81": "primary hadron from ministring into one hadron",
    "82": "primary hadron from ministring into two hadrons",
    "83": "primary hadron from normal string, fragmented off from the top of the string system",
    "84": "primary hadron from normal string, fragmented off from the bottom of the string system",
    "85": "primary produced hadrons in junction fragmentation of the first two string legs into the junction",
    "86": "primary produced hadrons in junction fragmentation of the second two string legs into the junction",
    "87": "primary produced baryon from a junction using standard junction fragmentation",
    "88": "primary produced baryon from a junction using the pearl-on-a-string gluon approximation",
    "89": "primary produced baryon from a junction in the ministring framework",
    "91": "normal decay products",
    "92": "decay products after oscillation B0 ↔ B0bar or B_s0 ↔ B_s0bar",
    "93": "decay handled by an external program, normally without oscillation",
    "94": "decay handled by an external program, with oscillation",
    "95": "a forced decay handled by an external program, normally without oscillation",
    "96": "a forced decay handled by an external program, with oscillation",
    "97": "decay products from a resonance produced in rescattering",
    "99": "particles with momenta shifted by Bose-Einstein effects",
}
