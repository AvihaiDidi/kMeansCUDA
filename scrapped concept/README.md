# FAKmeansCuda

This class is a scrapped idea for a Fast Approximate (FA) version of the k-means algorithem.
The idea was to get rid of all conditional statements in the algorithem, and create a version where the effect strength on each centroid is based not on which centroid is closest, but on the euclidian distance between each pixel and each centroid.
While I still think the idea has some merit, and does run faster then the normla implementation, in practice is just pushes the image towards it's average color.

I'm setting it aside here for now to focus on other projects, might return to it at some point later.