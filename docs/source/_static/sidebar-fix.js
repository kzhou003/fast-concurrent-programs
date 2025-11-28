/* Remove anchor links from sidebar - keep only document links */
document.addEventListener('DOMContentLoaded', function() {
    // Find all toctree-l1 items with anchor links
    const sidebarLinks = document.querySelectorAll('.wy-menu li.toctree-l1 > a[href*="#"]');
    sidebarLinks.forEach(function(link) {
        const li = link.parentElement;
        if (li) {
            li.style.display = 'none';
        }
    });
});
