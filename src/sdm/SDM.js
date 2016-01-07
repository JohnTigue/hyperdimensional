/* Main object type.
 * Parameters:
 *   size: # of "neurons" defaults to 10,000 as that is the commonly accepted place to start in order for the math to work
 */

/* While-designing notes:
 * The goal here is to see if JS can be used for neural nets without sacrificing performance.
 * Can various tricks be used to get the hardcore math to be done for JS via the GPU, or can something such as asm.js do the trick?
 *
 * ML libs:
 * http://www.datasciencecentral.com/profiles/blogs/machine-learning-in-javascript-a-compilation-of-resources
 * http://www.datascienceweekly.org/data-scientist-interviews/training-deep-learning-models-browser-andrej-karpathy-interview
 *
 * GPU libs:
 * https://github.com/kashif/node-cuda
 * http://stack.gl/
 *   via http://stackoverflow.com/questions/15213216/accessing-gpu-via-web-browser
 */
var SDM = (size = 10000) => {
  this.size = size;
  }
