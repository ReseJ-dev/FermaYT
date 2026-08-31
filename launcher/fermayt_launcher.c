#include <stdio.h>
#include <unistd.h>

int main(void) {
    execl(
        "/usr/bin/x-terminal-emulator",
        "x-terminal-emulator",
        "-e",
        "/home/grindyun/coding/FermaYT/start_fermayt.sh",
        (char *)NULL
    );

    perror("Unable to start FermaYT");
    return 1;
}
