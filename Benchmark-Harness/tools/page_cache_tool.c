// page_cache_tool: report how much of each file is resident in the OS page cache, and
// optionally evict it (posix_fadvise POSIX_FADV_DONTNEED), so a following model load is a
// genuine cold read from storage. No root needed: it only needs read access to the file, so
// for app-private model files run it as the app's uid via `run-as <package>`.
//
// Clean, unmapped pages are dropped; pages still mapped by a live process are not, so the
// app must be force-stopped first.
//
//   page_cache_tool [--evict] <file>...
// prints one line per file: "<path> resident_before=<pct> resident_after=<pct> bytes=<n>"
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static double resident_pct(int fd, off_t size) {
    if (size == 0) return 0.0;
    void *map = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) return -1.0;
    long page = sysconf(_SC_PAGESIZE);
    size_t pages = (size + page - 1) / page;
    unsigned char *vec = malloc(pages);
    double pct = -1.0;
    if (vec && mincore(map, size, vec) == 0) {
        size_t resident = 0;
        for (size_t i = 0; i < pages; i++) resident += vec[i] & 1;
        pct = 100.0 * resident / pages;
    }
    free(vec);
    munmap(map, size);
    return pct;
}

int main(int argc, char **argv) {
    int evict = 0, status = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--evict") == 0) { evict = 1; continue; }
        int fd = open(argv[i], O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) {
            printf("%s error=cannot_open\n", argv[i]);
            status = 1;
            if (fd >= 0) close(fd);
            continue;
        }
        double before = resident_pct(fd, st.st_size);
        if (evict) {
            fdatasync(fd);
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        }
        double after = resident_pct(fd, st.st_size);
        printf("%s resident_before=%.1f resident_after=%.1f bytes=%lld\n",
               argv[i], before, after, (long long) st.st_size);
        close(fd);
    }
    return status;
}
