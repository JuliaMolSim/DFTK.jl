module FourierTransforms

using AbstractFFTs

import AbstractFFTs: Plan, ScaledPlan,
                     fft, ifft, bfft, fft!, ifft!, bfft!,
                     plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
                     rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft,
                     fftshift, ifftshift,
                     rfft_output_size, brfft_output_size,
                     plan_inv, normalization

import Base: show, summary, size, ndims, length, eltype,
             *, inv, \

import LinearAlgebra: mul!

##############################################################################
# Native Julia FFTs:

include("ctfft.jl")

end
