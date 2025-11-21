// Initialize Mermaid with Material theme support
document$.subscribe(() => {
  mermaid.initialize({
    startOnLoad: true,
    theme: document.body.getAttribute('data-md-color-scheme') === 'slate' ? 'dark' : 'default',
    themeVariables: {
      primaryColor: '#5c6bc0',
      primaryTextColor: '#fff',
      primaryBorderColor: '#5c6bc0',
      lineColor: '#757575',
      secondaryColor: '#9fa8da',
      tertiaryColor: '#e8eaf6'
    },
    flowchart: {
      curve: 'basis',
      padding: 15
    },
    securityLevel: 'loose'
  });

  // Re-render Mermaid diagrams when theme changes
  const observer = new MutationObserver(() => {
    const scheme = document.body.getAttribute('data-md-color-scheme');
    mermaid.initialize({
      theme: scheme === 'slate' ? 'dark' : 'default'
    });

    // Find all mermaid elements and re-render
    document.querySelectorAll('.mermaid').forEach((element, index) => {
      const id = `mermaid-${index}`;
      const graphDefinition = element.textContent;
      element.removeAttribute('data-processed');
      element.innerHTML = graphDefinition;
      mermaid.init(undefined, element);
    });
  });

  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ['data-md-color-scheme']
  });
});
