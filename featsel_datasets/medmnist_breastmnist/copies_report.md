# Finding the copied images in BreastMNIST

A part of preparing the BreastMNIST dataset. The final report uses its
result: section 2.2 of
[`final_report/02_the_data.md`](../../final_report/02_the_data.md).

## The problem

Pawłowska, Karwat and Żołek (2023) checked BUSI, the source of
BreastMNIST's 780 images. Many images are copies of others: the same scan,
shifted, scaled or cropped. Some copies carry a different label. Their list
holds 153 groups of 346 images.

A random split can put one copy in the training part and its twin in the
test part. The model is then tested on an image it has seen. So the copies
must be known before splitting.

The letter names images by their BUSI file names. BreastMNIST does not keep
them. So the list cannot be mapped onto BreastMNIST's images.

## The method

We searched BreastMNIST's own images with the letter's method. Two images
are copies when one geometric transform maps one onto the other.

1. **Candidates.** Each image is checked against its 30 nearest images by
   ResNet-50 embedding.
2. **Matched points.** SIFT keypoints are matched between the two images.
   A match is kept only when it is clearly closer than the next best
   (Lowe's ratio, 0.75).
3. **One transform.** RANSAC fits one affine transform to the matches. At
   least 15 matches must agree with it, within 2 pixels.
4. **Pixels.** The first image is warped onto the second. The two must
   overlap on at least a fifth of the image. Over the overlap, their pixels
   must correlate above 0.9.
5. **Groups.** Copies join into groups through shared images: if A copies
   B and B copies C, all three are one group.

The separation is clear. 218 pairs had 50 or more agreeing points and a
correlation above 0.98. Almost every other pair had fewer than 10.
Checking 100 neighbours instead of 30 found no more.

## The result

| | The letter's list | Our search |
|---|---|---|
| Groups of copies | 153 | 150 |
| Images in them | 346 | 339 |
| Images beyond the first of each group | 193 | 189 |
| Groups mixing malignant with benign or normal | 6 | 6 |

The two agree closely. A few copies may be mirrored, which SIFT keypoints
do not match.

## How the copies differ

The search confirmed 230 pairs of copies.

- 1 pair is identical pixel for pixel.
- 27 more are moved by less than one pixel and scaled by less than 1%,
  yet their pixels differ.
- The other 202 are shifted or scaled. The median shift is 2 pixels and
  the largest is 28. The scale runs from 0.86 to 1.09.

As they stand, a copy differs from its twin by a median of 11 grey levels
out of 255. The images were cropped and resized separately, so the same
scan gives different pixels. Once aligned by the transform found, the two
match closely.

## Why a simpler test would not do

A search for identical images finds 1 pair of 230. The next simplest test
correlates two images as they are, without aligning them. Over all pairs of
the 780 images:

| Correlation of the images as they are | Copy pairs above it (of 230) | Other pairs above it (of 303,564) |
|---|---|---|
| 0.9 | 175 | 17 |
| 0.955, the highest of any other pair | 89 | 0 |

No threshold separates the two. At 0.9 it misses 55 copies and flags 17
pairs the search did not call copies. A threshold that flags none of those
misses 141 copies. Aligning the images first is what separates them.

## MedMNIST's own split leaks too

MedMNIST ships a fixed train, validation and test split. 90 of the 150
groups span more than one of its parts. 57 put copies in both its training
and its test part. One such pair, `train_115` and `test_076`, is identical
pixel for pixel, yet one is labelled malignant and the other benign or
normal.

## What the loader does

- `BreastMNIST()` runs the search on first use, in about 40 seconds. It
  caches the groups as `copies.csv` beside the downloaded images. The file
  is not tracked by git, like the images.
- Each image's group is the column `copy_group` of `data.samples`. It is
  missing for an image without copies.
- `BreastMNIST(find_copies=False)` skips the search and leaves the column
  out.
- `load_xy(drop_copies=True)` keeps the first image of each group whose
  labels agree. It drops each group whose labels conflict, since its true
  label is unknown. 585 images remain: 180 malignant and 405 benign or
  normal.

The search needs OpenCV, in the `embeddings` extra. Reading the cached
groups does not.

## Limits

- Mirrored copies are not found.
- A different OpenCV version could find slightly different groups. We
  accept this: the separation above leaves a wide margin.
- The 70 images of the axilla and the 7 with a biopsy needle, also
  reported by the letter, are not handled.

## Files

- [`copies.py`](copies.py): the search.
- [`breastmnist.py`](breastmnist.py): runs it and caches the groups.
- [`demos/copies.ipynb`](demos/copies.ipynb): the numbers above, with
  example pairs, the first image warped onto the second.

## Reference

Pawłowska, A., Karwat, P. and Żołek, N. (2023). Letter to the Editor. Re:
"Dataset of breast ultrasound images by W. Al-Dhabyani et al., Data in
Brief, 2020, 28, 104863". *Data in Brief* 48, 109247.
https://doi.org/10.1016/j.dib.2023.109247
