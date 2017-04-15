#ifndef __AESB_H__
#define __AESB_H__

void aesb_single_round(const uint8_t *in, uint8_t*out, const uint8_t *expandedKey);
void aesb_pseudo_round_mut(uint8_t *val, const uint8_t *expandedKey);

#define fast_aesb_single_round     aesb_single_round
#define fast_aesb_pseudo_round_mut aesb_pseudo_round_mut

#endif /* __AESB_H__ */
