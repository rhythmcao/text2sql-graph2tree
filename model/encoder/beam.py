from __future__ import division
import torch

class Beam(object):
    """
    Takes care of beams, back pointers, and scores.
    @args:
        size (int): beam size
        length (int): target sequence length
        device (torch.device)
    """

    def __init__(self, size, length, device=None):
        self.size = size
        self.length = length
        self.device = device
        # The score for each translation on the beam
        self.scores = torch.zeros(size, dtype=torch.float, device=self.device)
        # The backpointers at each time-step.
        self.prev_ks = []
        # The outputs at each time-step.
        self.next_ys = [torch.zeros(size, dtype=torch.long, device=self.device).fill_(0)]

    def advance(self, logprobs):
        """
        Given prob over BIO labels at the current timestep.
        Parameters:
            * `logprobs`- probs of advancing from the last step (3)
        Returns: True if beam search is complete.
        """
        assert logprobs.size(-1) == 3 # only B/I/O 3 labels
        if len(self.prev_ks) == 0: # the first timestep
            beam_scores = logprobs
        else: # broadcast, size x 3, attention that size may not equal to beam_size
            beam_scores = self.scores.unsqueeze(-1) + logprobs
        cur_total_num = beam_scores.numel() # number of elements
        top_k = self.size if cur_total_num > self.size else cur_total_num
        flatten_scores = beam_scores.contiguous().view(-1)
        best_scores, best_scores_id = flatten_scores.topk(top_k, 0, True, True)

        prev_k = best_scores_id // 3
        self.prev_ks.append(prev_k)
        next_y = best_scores_id - 3 * prev_k
        self.next_ys.append(next_y)
        self.scores = best_scores
        return self.done


    @property
    def done(self):
        return len(self.prev_ks) == self.length


    def get_hyps(self):
        num_hyps = self.scores.size(0)
        all_hyps = []
        for k in range(num_hyps):
            pointer, hyp = k, []
            for j in range(len(self.prev_ks) - 1, -1, -1):
                hyp.append(self.next_ys[j + 1][pointer].item())
                pointer = self.prev_ks[j][pointer].item()
            all_hyps.append(hyp[::-1])
        return all_hyps
