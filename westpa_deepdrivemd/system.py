import numpy 
import west
import os
from west import WESTSystem
from westpa.binning import RectilinearBinMapper 
from westpa.binning import FuncBinMapper 
from westpa.binning import RecursiveBinMapper 
import logging

log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

class SDASystem(WESTSystem):
    '''
    Class specify binning schemes, walker counts, and other core weighted
    ensemble parameters.
    '''

    def initialize(self):

        # pcoord dimensionality and length
        self.pcoord_ndim            = 2
        self.pcoord_len             = 11 # (nstlim / ntwx + 1) #11 # 3
        self.pcoord_dtype           = numpy.float32

        # Latent space has a standard normal prior so we can expect most of
	# the laten coordinates to be within (-3, 3). Using 0.5 space gives 12x12 grid + infinities
        latent_dim = 2
        space = ["-inf"] + list(numpy.arange(-3.0, 3.0, 0.5)) + ["inf"]
        output_space = [space.copy() for _ in range(latent_dim)]
        self.bin_mapper = RectilinearBinMapper(output_space)

        self.bin_target_counts = numpy.empty(
            (self.bin_mapper.nbins,), dtype=numpy.int
        )

        # Number of walkers (trajectories) per bin
        self.bin_target_counts[...] = 8 # 1
        # set to 1 for testing try 2 if it breaks. number of target trajs per bin.. 
        # 1 walker per bin means fewer trajectories. 8

    def test_initialize(self):

#       pcoord dimensionality and length
        self.pcoord_ndim            = 2
        self.pcoord_len             = 3 # 11
        self.pcoord_dtype           = numpy.float32
        
#       These are your "outer" bins in both dimensions
        outer_mapper = RectilinearBinMapper(
                [[0, 6.5, 8.9, 'inf'],
                 [0, 'inf']]
                                            )
#       These are your "inner" bins in both dimensions      
        inner_mapper = RectilinearBinMapper(
                [[6.5, 8.9,'inf'],
                 [0, 1, 2, 3, 3.5, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,  5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,  5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,6.8, 6.9, 7.0, 7.1, 7.3, 7.5, 7.6, 7.65,  7.7, 7.75,  7.8, 7.85, 7.9, 7.95, 8.0, 8.05, 8.1, 8.15, 8.2, 8.25, 8.3, 8.35, 8.4, 8.45, 8.5, 8.6, 8.7, 8.9, 9.1, 9.3, 9.5, 9.7, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5,'inf']]
                                            )
        
        # Testing. If this has erros, make it [0, 2, .., 12, 'inf]
        #inner_mapper = RectilinearBinMapper([[6.5, 8.9,'inf'], [0, 2, 'inf']])
         
        self.bin_mapper = RecursiveBinMapper(outer_mapper)

#       This coordinate in both dimensions specifies which "outer" bin to place your "inner" bin into
        self.bin_mapper.add_mapper(
                inner_mapper, 
                [6.6,1]
                                   )

        self.bin_target_counts = numpy.empty((self.bin_mapper.nbins,), 
                                             dtype=numpy.int)

#       Number of walkers per bin
        self.bin_target_counts[...] = 1 
        # set to 1 for testing try 2 if it breaks. number of target trajs per bin.. 1 walker per bin means fewer trajectories. 8
