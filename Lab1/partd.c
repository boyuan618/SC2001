#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// Global counter for comparisons
long long comparisons = 0;

// Merge function
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        comparisons++;
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

// Pure merge sort
void merge_sort(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        merge_sort(arr, l, m);
        merge_sort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 7) {
        if (rank == 0) fprintf(stderr, "Error: need exactly 7 ranks (got %d)\n", size);
        MPI_Finalize();
        return 1;
    }

    long test_sizes[] = {1000, 2000, 5000, 10000, 100000, 1000000, 10000000};
    long n = test_sizes[rank];

    // Allocate array
    int *arr = malloc(n * sizeof(int));
    if (!arr) {
        fprintf(stderr, "Rank %d: malloc failed for n=%ld\n", rank, n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Generate array
    srand(42 + rank);
    for (long i = 0; i < n; i++) arr[i] = rand() % n;

    // Reset comparisons
    comparisons = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    merge_sort(arr, 0, n - 1);
    double end = MPI_Wtime();
    double elapsed = end - start;

    // Print CSV result
    printf("%ld,%lld,%.6f\n", n, comparisons, elapsed);

    free(arr);
    MPI_Finalize();
    return 0;
}
