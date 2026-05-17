#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared waveform catalog used by benchmark scripts."""

TRI_ASYMMETRIES = [0.20, 0.35, 0.50, 0.65, 0.72, 0.80]

SHAPES = {
    'sine':             ('Sine',                                '#4488FF'),
    'rect':             ('Square',                              '#888888'),
    'tri_sym':          ('Triangle\n(symmetric, 0.50)',        '#FF8800'),
    'saw_up':           ('Sawtooth\n(up, 0% dwell)',           '#AAAAAA'),
    'saw_down':         ('Sawtooth\n(down, 0% dwell)',         '#CCCCCC'),
    'tri_20':           ('Triangle\n(asym 0.20)',              '#6d1f80'),
    'tri_35':           ('Triangle\n(asym 0.35)',              '#982c80'),
    'tri_50':           ('Triangle\n(asym 0.50)',              '#bc3978'),
    'tri_65':           ('Triangle\n(asym 0.65)',              '#d8576b'),
    'tri_72':           ('Triangle\n(asym 0.72)',              '#e9775a'),
    'tri_80':           ('Triangle\n(asym 0.80)',              '#f89c46'),
    'gaussian':         ('Gaussian\nBiphasic',                 '#00CCAA'),
    'raised_cos':       ('Raised\nCosine',                     '#FF44AA'),
    'exponential':      ('Exponential\nBiphasic',              '#FF6622'),
    'half_wave':        ('Half-wave\nRectified Sine',          '#22DDFF'),
    'sinc':             ('Sinc\nPulse',                        '#BB44FF'),
    'staircase_4':      ('Staircase\n(4-step)',                '#FFCC00'),
    'trapezoid_flat35': ('Trapezoidal\n(flat 35%)',            '#66FF88'),
    'chirp':            ('Linear\nChirp',                      '#FF2288'),
    'prbs':             ('PRBS\nBiphasic',                     '#AAFFCC'),
    'biomimetic_ap':    ('Biomimetic\nAP-shape',               '#00FF88'),
    'cathodic_first':   ('Cathodic-First\nInterleaved',        '#FF8844'),
    'anodic_first':     ('Anodic-First\nInterleaved',          '#4488FF'),
    'double_pulse':     ('Double\nPulse',                      '#DDAAFF'),
    'composite_harm':   ('Composite\nHarmonics',               '#FFEE44'),
    'soft_clip':        ('Soft-Clipped\nSine',                 '#88FFCC'),
    'erf_sigmoid':      ('ERF Sigmoid\nBiphasic',              '#FF44CC'),
}
