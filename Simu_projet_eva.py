#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 23:27:28 2025

@author: hjacinto

To use the file go in the main section at the very bottom !
"""

from collections import namedtuple
import scipy
from tqdm import tqdm
import os
import stim
import sinter
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from Coordinates import Coord, SPLIT_DIST
from noise_v2 import NoiseModel, Probas

# %% Build the circuit


def multiply_namedtuple(namedtuple_instance, multiplier):
    """Multiply a namedtuple by a float."""
    return namedtuple_instance._replace(
        **{field: getattr(namedtuple_instance, field) * multiplier for field in namedtuple_instance._fields})


def sum_namedtuples(*tuples):
    """Sum several named tuples."""
    if not tuples:
        return None

    TupleType = type(tuples[0])
    fields = tuples[0]._fields

    summed_values = [sum(getattr(t, field) for t in tuples) for field in fields]

    return TupleType(*summed_values)


def merge_dicts(dict1, dict2):
    """Merge of dict appropriate for dict of stabs or qubits coordinates."""
    result = {}
    # Loop through keys in both dictionaries
    for key in dict1.keys() | dict2.keys():
        # Concatenate lists if key exists in both dictionaries
        result[key] = dict1.get(key, []) + dict2.get(key, [])

    return result


def flatten(iterable_of_lists):
    """Flatten a list of lists (or any iterable of lists)."""
    return sum(iterable_of_lists, [])


def myfunction(progress):
    """For the progress bar."""
    clear_output(wait=True)
    print(progress.status_message)


def flatten_dict(dico):
    """Make a list out of the values of a dictionnary that should be lists."""
    return sum((v for v in dico.values()), [])


def plot_stabs(data_qubits, x_stabs, z_stabs, convention='ij'):
    """Representation of where are the data qubits and stabilizers.

    The input format is the output from surf_qubits_stabs (and other functions).
    Normal convention is 'ij', but as crumble uses the 'xy' convention, it is
    allowed to use it (but note that I changed the coordinates of the qubits in
    stim so that crumble also shows as following the 'ij' convention).
    """
    if convention not in ('ij', 'xy'):
        raise ValueError("Convention must be 'ij' or 'xy'!")
    _, ax = plt.subplots()
    data = np.array(data_qubits)
    x = np.array(flatten_dict(x_stabs))
    z = np.array(flatten_dict(z_stabs))
    if convention == 'ij':  # invert x and y axis
        data = np.flip(data, axis=1)
        x = np.flip(x, axis=1)
        z = np.flip(z, axis=1)
    ax.scatter(data[:, 0], data[:, 1], color='red', edgecolors='black',
               label="data")
    ax.scatter(x[:, 0], x[:, 1], color='grey', edgecolors='black', label="x")
    ax.scatter(z[:, 0], z[:, 1], color='white', edgecolors='black', label="z")
    if convention == 'ij':
        ax.invert_yaxis()
        ax.xaxis.tick_top()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_aspect('equal', 'box')
    ax.grid(linestyle=':')
    ax.set_axisbelow(True)
    plt.show()


def nbr_cycle(d):
    """Return the number of cycle in one simulation."""
    return (3*d)


def stat_error(p, n, rep=1):
    """Statistical error for a binomial sampling."""
    return (np.sqrt(p*(1-p)/n)/(rep*(1-p)**(1-1/rep)))


def to_error_per_round(proba, nb):
    """Transform the logical error rate after d rounds to the logical error rate per rounds."""
    return (1-(1-proba)**(1/nb))

# %% Stabilizer description


def surf_qubits_stabs(dist_i, dist_j=None, shift=(0, 0)):
    """Génère les qubits et stabilisateur d'un qubit logique."""
    if dist_j is None:
        dist_j = dist_i
    data_qubits = [Coord(2*i, 2*j)
                   for i in range(1, dist_i+1) for j in range(1, dist_j+1)]
    x_stabs = {
        'top': [Coord(1, 1+2*j) for j in range(1, dist_j) if j % 2 == 0],
        'bottom': [Coord(1+2*dist_i, 1+2*j) for j in range(1, dist_j)
                   if (j+dist_i) % 2 == 0],
        'bulk': [Coord(1+2*i, 1+2*j) for i in range(1, dist_i)
                 for j in range(1, dist_j) if (i+j) % 2 == 0]
    }
    z_stabs = {
        'left': [Coord(1+2*i, 1) for i in range(1, dist_i) if i % 2 == 1],
        'right': [Coord(1+2*i, 1+2*dist_j) for i in range(1, dist_i)
                  if (i+dist_j) % 2 == 1],
        'bulk': [Coord(1+2*i, 1+2*j) for i in range(1, dist_i)
                 for j in range(1, dist_j) if (i+j) % 2 == 1]
    }
    # Check that only (even, even) and (odd, odd) coordinates are possible
    assert all((x[0] + x[1]) % 2 == 0 for x in data_qubits)
    assert all((x[0] + x[1]) % 2 == 0
               for s in {**x_stabs, **z_stabs}.values() for x in s)
    # Shift the logical qubit
    data_qubits = [x + shift for x in data_qubits]
    x_stabs = {key: [x + shift for x in val] for key, val in x_stabs.items()}
    z_stabs = {key: [x + shift for x in val] for key, val in z_stabs.items()}
    return data_qubits, x_stabs, z_stabs


def _prepare_ids(data_qubits, x_stabs, z_stabs):
    """Prepare metadata for easier handling of indexes.

    qubits_dict and qubits_id_dict are guaranted to be in the same order.
    """
    qubits_dict = {'data': data_qubits,
                   'x': flatten_dict(x_stabs), 'z': flatten_dict(z_stabs)}
    # All the qubits and the indexes
    qubits_list = sorted(flatten_dict(qubits_dict))
    qubits_index = {q: i for i, q in enumerate(qubits_list)}
    # Add for convenience the set of all stabilizers qubits ; main interest is
    # that it's lenght allow to easily compute where the results are in the
    # measurement reg.
    qubits_dict['stabs'] = sum(
        (qubits_dict[i] for i in ('x', 'z')), [])
    # Indexes for each type of qubits.
    qubits_id_dict = {key: [qubits_index[q] for q in val]
                      for key, val in qubits_dict.items()}
    assert all(qubits_dict[key] == [qubits_list[i] for i in qubits_id_dict[key]]
               for key in qubits_dict.keys()), "Order not coinciding."
    # Add virtual unsplit qubits for teleported measurements
    qubits_dict['stabs_virtual'] = sum(
        (qubits_dict[i] for i in ('x', 'z')), [])
    return qubits_list, qubits_index, qubits_dict, qubits_id_dict

# %% Circuit synthesis


def _initialize_circuit(state, qubits_list, qubits_id):
    """Create the circuit, and initialize it.

    state is either '0' or '+'
    """
    state = state.casefold()
    if state not in ('0', '+'):
        raise ValueError("State should be '0' or '+'!")
    # Create circuit and label qubits
    circuit = stim.Circuit()
    for i, q in enumerate(qubits_list):
        circuit.append('QUBIT_COORDS', i, q.stim_coord())
    # Initialisation in |0>|0>|0>...|0> or |+>|+>|+>...|+>
    # circuit.append({'0': "R", '+': "RX"}[state], qubits_id['data'])
    # We consider we can't prepare + :
    circuit.append("R", qubits_id['data'])
    # Initialisation in 0 of the stabilisers ancilary qubits
    circuit.append("R", qubits_id['stabs'])
    if state == '+':
        circuit.append("H", qubits_id['data'])
    return circuit


def _surface_code_cycle(dist_i, dist_j, data_qubits, x_stabs, z_stabs):
    """Un cycle de correction d'erreur, du H aux mesures/réinitialisation.

    Mais sans les détecteurs (car ils changent entre le premier cycle et les
    autres).
    """
    qubits_list, qubits_index, qubits, qubits_id = _prepare_ids(
        data_qubits, x_stabs, z_stabs)
    # Create the circuit
    circuit = stim.Circuit()

    circuit.append("TICK")
    # Prepare the ancillaes :
    # Prepare the X ancilla in the correct base.
    circuit.append("H", qubits_id['x'])  # No error as virtual gate

    # Note : we implement here https://arxiv.org/abs/1404.3747

    for dirr_x, dirr_z in zip([(-1, 1), (-1, -1), (1, 1), (1, -1)],
                              [(-1, 1), (1, 1), (-1, -1), (1, -1)]):
        cnot_args = []
        # Temporary circuit to identify iddle qubits
        circuit_temp = stim.Circuit()
        circuit_mes_temp = stim.Circuit()
        for x in qubits['stabs_virtual']:
            if x in qubits['z'] and x + dirr_z in data_qubits:
                cnot_args.extend([qubits_index[x + dirr_z], qubits_index[x]])
            elif x in qubits['x'] and x + dirr_x in data_qubits:
                cnot_args.extend([qubits_index[x], qubits_index[x + dirr_x]])
        circuit_temp.append("CX", cnot_args)
        circuit += circuit_temp
        circuit.append("TICK")
        circuit += circuit_mes_temp
    # Rotate basis because we only mesure in Z basis
    circuit.append("H", qubits_id['x'])
    # Do the measures
    circuit.append("TICK")
    circuit.append("MR", qubits_id['stabs'])
    return circuit


def _add_surface_code_detectors(bloc, qubits, nb_stabs, x=True, z=True,
                                first=False, time=0):
    """Add detectors of the surface code.

    Assumes that the stabilizers where measured in the orders of qubits['stabs']
    """
    for s in (qubits['x'] if x else []) + (qubits['z'] if z else []):
        mes = [stim.target_rec(qubits['stabs'].index(s) - nb_stabs)]
        if not first:  # Comparison with last turn.
            mes += [stim.target_rec(qubits['stabs'].index(s) - 2*nb_stabs)]
        bloc.append("DETECTOR", mes, s.stim_coord(time))


def _add_final_measure(circuit, kind, nb_data, nb_stabs, qubits, qubits_id,
                       time=1, split_col=None):
    """Add the final measurements and corresponding detectors."""
    kind = kind.casefold()
    if kind not in ('x', 'z'):
        raise ValueError("'kind' must be 'x' or 'z'!")
    circuit.append("TICK")
    circuit.append({'x': "MX", 'z': 'M'}[kind], qubits_id['data'])
    # Vérification des stabiliseurs à la main après mesure
    for s in qubits[kind]:
        circuit.append(
            "DETECTOR",
            [stim.target_rec(qubits['data'].index(s + dirr) - nb_data)
             for dirr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
             if s + dirr in qubits['data']] +
            [stim.target_rec(qubits['stabs'].index(s) - nb_stabs - nb_data)],
            s.stim_coord(time))


def gen_memory(dist_i, dist_j, repeat,
               probas, kind='x', plot=False):
    """Génère le circuit pour une mémoire où on stoque |+> ou |0>.

    kind : 'x' or 'z'
    """
    kind = kind.casefold()
    if kind not in ('x', 'z'):
        raise ValueError("'kind' must be 'x' or 'z'!")
    if repeat is None:
        repeat = dist_i
    # Prepare qubits and stabilizers
    data_qubits, x_stabs, z_stabs = surf_qubits_stabs(dist_i, dist_j)
    # Choose between old version and new splitting
    qubits_list, qubits_index, qubits, qubits_id = _prepare_ids(
        data_qubits, x_stabs, z_stabs)
    if plot:
        plot_stabs(data_qubits, x_stabs, z_stabs,
                   convention='ij')

    # Useful for computing indexes in measurement record.
    nb_data, nb_stabs = len(qubits['data']), len(qubits['stabs'])
    # Initialize circuit and qubits
    circuit = _initialize_circuit({'x': '+', 'z': '0'}[kind],
                                  qubits_list, qubits_id)
    # First cycle
    circuit += _surface_code_cycle(dist_i, dist_j, data_qubits, x_stabs, z_stabs)
    _add_surface_code_detectors(circuit, qubits, nb_stabs,
                                x=(kind == 'x'), z=(kind == 'z'), first=True)
    # Generic cycle
    bloc = _surface_code_cycle(dist_i, dist_j, data_qubits, x_stabs, z_stabs)
    bloc.append("SHIFT_COORDS", [], (0, 0, 1))
    _add_surface_code_detectors(bloc, qubits, nb_stabs)
    circuit.append(stim.CircuitRepeatBlock(repeat, bloc))
    # Mesure finale et assertion résultat
    _add_final_measure(circuit, kind, nb_data, nb_stabs, qubits, qubits_id,
                       time=1)
    # Observable
    if kind == 'x':
        logic_pauli = [Coord(2*i, 2) for i in range(1, dist_i+1)]
    elif kind == 'z':
        logic_pauli = [Coord(2, 2*j) for j in range(1, dist_j+1)]
    else:
        raise RuntimeError("Nothing to do here")
    circuit.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(qubits['data'].index(x) - nb_data)
                    for x in logic_pauli], 0)
    with open("circuits/circuit.html", 'w') as file:
        print(circuit.diagram("interactive"), file=file)
    # Prepare the list of qubits involved in bell pairs
    noisy_circuit = NoiseModel.Standard(probas).noisy_circuit(circuit,
                                                              auto_push_prep=False,
                                                              auto_pull_mes=False,
                                                              auto_push_prep_bell=True,
                                                              auto_pull_mes_bell=True,
                                                              bell_pairs=[]
                                                              )
    # slow_prep = True means that two idle are applied during bell pair prep otherwise only one
    # idle is applied
    # with open('circuits/diagram_debug.svg', 'w') as f:
    #    print(circuit.diagram("timeline-svg"), file=f)$
    # Print the circuit in a file to check error model
    if dist_i == 3:
        with open('circuits/diagram_with_error.svg', 'w') as f:
            print(noisy_circuit.diagram("timeline-svg"), file=f)
    return noisy_circuit
# %% Build the task and collect the Monte Carlo simulations


def generate_tasks(kind, rep, probas):
    """Generate surface code circuit without splitting tasks using Stim's circuit generation."""
    # As a reminder, Probas=['Hadam','idle_data','idle_bell','depol', 'prep', 'mes', 'bell']
    for d in tqdm([3, 5, 7, 9, 11], desc='Inner loop', leave=False):
        yield sinter.Task(
            circuit=gen_memory(d, d, rep(d), probas, kind, None),
            json_metadata={
                'd': d,
                'k': rep(d)
            },
        )


def _collect_and_print(tasks, show_table=True,
                       file=None,
                       read_file=None,
                       **kwargs):
    """Collect and print tasks, kwargs is passed to sinter.collect."""
    nb_shots = 3_000_000
    if read_file is not None:
        if os.path.exists(read_file):
            print('This data file already exist')
            kwargs = dict(num_workers=8, existing_data_filepaths=[read_file], max_shots=nb_shots,
                          max_errors=10000, tasks=tasks, decoders=['pymatching'], print_progress=True) | kwargs
        else:
            print('Create a new data file')
            kwargs = dict(num_workers=8, save_resume_filepath=read_file, max_shots=nb_shots, max_errors=1000,
                          tasks=tasks, decoders=['pymatching'], print_progress=True) | kwargs
    elif file is not None:
        print('Create a new data file')
        kwargs = dict(num_workers=8, save_resume_filepath=file, max_shots=nb_shots, max_errors=1000,
                      tasks=tasks, decoders=['pymatching'], print_progress=True) | kwargs
    else:
        kwargs = dict(num_workers=8, max_shots=nb_shots, max_errors=1000,
                      tasks=tasks, decoders=['pymatching'], print_progress=True) | kwargs
    # ,progress_callback=myfunction can be added
    # Collect the samples (takes a few minutes).
    samples = sinter.collect(**kwargs)

    if show_table:
        # Print samples as CSV data.
        print(sinter.CSV_HEADER)
        for sample in samples:
            print(f"p_l={sample.errors/sample.shots:e},\t" +
                  ',\t'.join(f"{k}={v}" for k, v in sorted(
                      sample.json_metadata.items())))
# Do the plot
    return samples


def _plot_per_round(samples: list[sinter.TaskStats],
                    x_axis='d', x_label="Distance", label="d={d}",
                    title="Surface code", ylim=None, filename=None, filtered=False, rep=nbr_cycle):
    """Fait le travail de dessiner avec un taux d'erreur logique par round.

    label est formaté à partir du dictionnaire stat.json_metadata
    """
    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    if filtered:
        sinter.plot_error_rate(
            ax=ax,
            stats=samples,
            group_func=lambda stat: label.format_map(stat.json_metadata),
            filter_func=lambda stat: (stat_error(stat.errors/stat.shots, stat.shots, rep=stat.json_metadata['k']) < to_error_per_round(
                stat.errors/stat.shots, stat.json_metadata['k'])/4),

            x_func=lambda stat: stat.json_metadata[x_axis],
            failure_units_per_shot_func=lambda stats: stats.json_metadata['k']
            # highlight_max_likelihood_factor = 1
        )
    else:
        sinter.plot_error_rate(
            ax=ax,
            stats=samples,
            group_func=lambda stat: label.format_map(stat.json_metadata),
            x_func=lambda stat: stat.json_metadata[x_axis],
            failure_units_per_shot_func=lambda stats: stats.json_metadata['k']
            # highlight_max_likelihood_factor = 1
        )
    ax.loglog()
    ax.set_ylim(ylim)
    ax.grid()
    ax.set_title(title)
    ax.set_ylabel("Logical error probability")
    ax.set_xlabel(x_label)
    ax.legend()
    # Save to file and also open in a window.
    if filename is not None:
        fig.savefig('figures/'+filename)
    plt.show()
    return fig, ax


# %% function to call
def surface_code_scaling(probas, kind='x', rep=nbr_cycle, filtered=True):
    """Code de surface normal."""
    samples = _collect_and_print(generate_tasks(kind, rep, probas))
    # Plot the logical error rate per cycle
    _plot_per_round(samples, title=r"  ",
                    filename="surface_code_threshold_per_round" + kind + ".pdf", filtered=filtered)


# %% Partie exécutable
# Hello Eva ! to use this fule you can adjust the parameters of the different noise right below (for
# the moment I fixed them at a first guess) and then you can run the file.
# It will give you a plot of the logical error rate per round of error correction as a function of
# the distance.
# If the logical error rate gets smaller and smaller when the distance increases then it means that
# you are below the threshold !
# Let me know if you need any new feature
if __name__ == '__main__':
    # Reminder that
    p_H, p_idle, pCNOT, p_reset, p_mes = 1e-3, 1e-4, 1e-3, 1e-3, 2e-2
    probas = Probas(p_H, p_idle, 0, pCNOT, p_reset, p_mes, 0)
    surface_code_scaling(probas)
