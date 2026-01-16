(() => {
  const searchInput = document.getElementById("session-search");
  const table = document.querySelector('[data-table="session"]');
  if (!searchInput || !table) {
    return;
  }

  const rows = Array.from(table.querySelectorAll("tbody tr"));

  const normalize = (value) => value.toLowerCase();

  searchInput.addEventListener("input", (event) => {
    const query = normalize(event.target.value.trim());
    rows.forEach((row) => {
      if (row.querySelector(".empty")) {
        row.style.display = "";
        return;
      }
      const text = normalize(row.textContent);
      row.style.display = text.includes(query) ? "" : "none";
    });
  });
})();

// Per-table horizontal pagination
function attachTableNav() {
  document.querySelectorAll('.table-wrap, .table-scroll-container').forEach((wrap) => {
    if (wrap.classList.contains('no-table-nav')) return;
    // Avoid duplicating controls
    if (wrap.previousElementSibling?.classList.contains('table-nav')) return;

    const nav = document.createElement('div');
    nav.className = 'table-nav';

    const prev = document.createElement('button');
    prev.className = 'btn table-nav-btn';
    prev.textContent = '←';
    prev.addEventListener('click', () => {
      const scrollAmount = wrap.clientWidth * 0.8;
      wrap.scrollLeft = wrap.scrollLeft - scrollAmount;
    });

    const next = document.createElement('button');
    next.className = 'btn table-nav-btn';
    next.textContent = '→';
    next.addEventListener('click', () => {
      const scrollAmount = wrap.clientWidth * 0.8;
      wrap.scrollLeft = wrap.scrollLeft + scrollAmount;
    });

    nav.appendChild(prev);
    nav.appendChild(next);
    
    // Insert nav BEFORE the table wrap
    wrap.parentNode.insertBefore(nav, wrap);
  });
}

// Client-side filtering for session/all matches table
function initSessionFilters() {
  const form = document.querySelector('.filters');
  if (!form) return;

  const selects = Array.from(form.querySelectorAll('select'));
  selects.forEach((sel) => sel.addEventListener('change', () => form.submit()));
}

// Generic table search filters
function initTableFilters() {
  document.querySelectorAll('[data-table-filter]').forEach((input) => {
    const tableName = input.dataset.tableFilter;
    if (!tableName) return;
    const table = document.querySelector(`[data-table="${tableName}"]`);
    if (!table) return;
    const rows = Array.from(table.querySelectorAll('tbody tr'));

    input.addEventListener('input', () => {
      const query = input.value.trim().toLowerCase();
      rows.forEach((row) => {
        if (row.querySelector('.empty')) {
          row.style.display = '';
          return;
        }
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
      });
    });
  });
}

// Theme Toggle
function toggleTheme() {
  const html = document.documentElement;
  const currentTheme = html.getAttribute('data-theme') || 'dark';
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  
  // Update toggle button emoji
  const toggle = document.querySelector('.theme-toggle');
  if (toggle) {
    toggle.textContent = newTheme === 'dark' ? '🌙' : '☀️';
  }
}

// Load saved theme on page load
(function() {
  const savedTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  
  // Update toggle button on load
  document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.querySelector('.theme-toggle');
    if (toggle) {
      toggle.textContent = savedTheme === 'dark' ? '🌙' : '☀️';
    }
  });
})();

// Export Dropdown
function toggleExport() {
  const menu = document.getElementById('exportMenu');
  if (menu) {
    menu.classList.toggle('show');
  }
}

// Mobile navigation toggle
function toggleNav() {
  const body = document.body;
  if (!body) return;
  if (body.classList.contains('nav-open')) {
    closeNav();
  } else {
    openNav();
  }
}

function openNav() {
  const body = document.body;
  if (!body) return;
  body.classList.add('nav-open');
  const toggle = document.querySelector('.nav-toggle');
  if (toggle) {
    toggle.setAttribute('aria-expanded', 'true');
  }
}

function closeNav() {
  const body = document.body;
  if (!body) return;
  body.classList.remove('nav-open');
  const toggle = document.querySelector('.nav-toggle');
  if (toggle) {
    toggle.setAttribute('aria-expanded', 'false');
  }
}

function initMobileNavGroups() {
  document.querySelectorAll('.site-nav .nav-link').forEach((link) => {
    link.addEventListener('click', () => {
      if (!window.matchMedia('(max-width: 980px)').matches) return;
      closeNav();
    });
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeNav();
    }
  });
}

function initStickyFilters() {
  document.querySelectorAll('form.filters').forEach((form) => {
    form.classList.add('sticky');
  });
}

function initFilterChips() {
  document.querySelectorAll('form.filters').forEach((form) => {
    const buildChips = () => {
      let chipContainer = form.nextElementSibling;
      if (!chipContainer || !chipContainer.classList.contains('filter-chips')) {
        chipContainer = document.createElement('div');
        chipContainer.className = 'filter-chips';
        form.insertAdjacentElement('afterend', chipContainer);
      }
      chipContainer.innerHTML = '';

      const inputs = Array.from(form.querySelectorAll('select, input'));
      const chips = [];

      inputs.forEach((input) => {
        const value = String(input.value || '').trim();
        if (!value || value === 'all') return;
        const label = input.closest('label');
        const labelText = label ? (label.childNodes[0]?.textContent || label.textContent || '').trim() : input.name;
        const defaultOption = input.tagName === 'SELECT' ? input.querySelector('option')?.value : '';
        const resetValue = input.tagName === 'SELECT' ? (defaultOption ?? '') : '';

        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'filter-chip';
        chip.textContent = `${labelText}: ${value}`;
        chip.addEventListener('click', () => {
          if (input.tagName === 'SELECT') {
            input.value = resetValue;
          } else {
            input.value = '';
          }
          form.submit();
        });
        chips.push(chip);
      });

      chips.forEach((chip) => chipContainer.appendChild(chip));

      if (chips.length) {
        const clear = document.createElement('button');
        clear.type = 'button';
        clear.className = 'filter-clear';
        clear.textContent = 'Clear Filters';
        clear.addEventListener('click', () => {
          inputs.forEach((input) => {
            if (input.tagName === 'SELECT') {
              const first = input.querySelector('option');
              if (first) input.value = first.value;
            } else {
              input.value = '';
            }
          });
          form.submit();
        });
        chipContainer.appendChild(clear);
      }
    };

    form.addEventListener('change', buildChips);
    buildChips();
  });
}

function initTableDensity() {
  const root = document.documentElement;
  const buttons = Array.from(document.querySelectorAll('.density-btn'));
  if (!buttons.length) return;

  const setDensity = (density) => {
    const value = ['compact', 'balanced', 'full'].includes(density) ? density : 'balanced';
    root.setAttribute('data-density', value);
    localStorage.setItem('tableDensity', value);
    buttons.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.density === value);
    });
  };

  const saved = localStorage.getItem('tableDensity') || 'balanced';
  setDensity(saved);

  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      setDensity(btn.dataset.density || 'balanced');
    });
  });
}

function initPlayerHoverCard() {
  if (!window.matchMedia('(hover: hover)').matches) return;
  const data = window.playerHoverData || {};
  const targets = Array.from(document.querySelectorAll('.player-name'));
  if (!targets.length) return;

  const card = document.createElement('div');
  card.className = 'player-hover-card';
  document.body.appendChild(card);

  let hideTimer = null;

  const hideCard = () => {
    card.classList.remove('visible');
  };

  const showCard = (target) => {
    const name = (target.dataset.player || target.textContent || '').trim();
    if (!name) return;
    const info = data[name.toLowerCase()];
    if (!info) return;

    card.innerHTML = `
      <div class="hover-name">${info.player}</div>
      <div class="hover-row"><span class="hover-label">Win %</span><span>${info.win_pct}</span></div>
      <div class="hover-row"><span class="hover-label">KDA</span><span>${info.kda}</span></div>
      <div class="hover-row"><span class="hover-label">CSR</span><span>${info.csr}</span></div>
      <div class="hover-row"><span class="hover-label">Last</span><span>${info.last_match}</span></div>
    `;

    card.style.left = '0px';
    card.style.top = '0px';
    card.classList.add('visible');

    const rect = target.getBoundingClientRect();
    const cardRect = card.getBoundingClientRect();
    const padding = 12;
    let left = rect.left + rect.width / 2 - cardRect.width / 2;
    left = Math.max(padding, Math.min(left, window.innerWidth - cardRect.width - padding));

    let top = rect.top - cardRect.height - 12;
    if (top < padding) {
      top = rect.bottom + 12;
    }
    card.style.left = `${left}px`;
    card.style.top = `${top}px`;
  };

  targets.forEach((target) => {
    target.addEventListener('mouseenter', () => {
      clearTimeout(hideTimer);
      showCard(target);
    });
    target.addEventListener('mouseleave', () => {
      hideTimer = setTimeout(hideCard, 120);
    });
  });
}

function updateCsrOnlineStatus() {
  const dots = document.querySelectorAll('[data-last-match]');
  if (!dots.length) return;
  const now = Date.now();
  dots.forEach((dot) => {
    const raw = dot.getAttribute('data-last-match');
    if (!raw) return;
    const ts = Date.parse(raw);
    if (Number.isNaN(ts)) return;
    const diffMinutes = (now - ts) / 60000;
    const isOnline = diffMinutes <= 20;
    dot.classList.toggle('online', isOnline);
    const rounded = Math.max(0, Math.round(diffMinutes));
    const statusLabel = isOnline ? 'Online' : 'Offline';
    dot.setAttribute('title', `${statusLabel} (last game ${rounded}m ago)`);
  });
}

function initCsrOnlineStatus() {
  updateCsrOnlineStatus();
  setInterval(updateCsrOnlineStatus, 60000);
}

// Close export dropdown when clicking outside
document.addEventListener('click', (e) => {
  const dropdown = document.querySelector('.export-dropdown');
  const menu = document.getElementById('exportMenu');
  if (dropdown && menu && !dropdown.contains(e.target)) {
    menu.classList.remove('show');
  }
});

// Keyboard navigation for horizontally scrollable tables
document.addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  const wrap = Array.from(document.querySelectorAll('.table-wrap')).find((el) => {
    const rect = el.getBoundingClientRect();
    return rect.top < window.innerHeight && rect.bottom > 0;
  });
  if (!wrap) return;
  const scrollAmount = wrap.clientWidth * 0.8;
  wrap.scrollLeft = wrap.scrollLeft + (e.key === 'ArrowLeft' ? -scrollAmount : scrollAmount);
});

function parseSortDate(text) {
  if (!text) return null;
  const hasDate = /\d{4}-\d{2}-\d{2}/.test(text) || /\d{1,2}\/\d{1,2}\/\d{2,4}/.test(text);
  if (!hasDate) return null;
  const parsed = Date.parse(text);
  return Number.isNaN(parsed) ? null : parsed;
}

function parseSortTime(text) {
  if (!text) return null;
  const trimmed = text.trim();
  const daysMatch = trimmed.match(/^(\d+)\s+days?\s+(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)$/);
  if (daysMatch) {
    const days = Number(daysMatch[1]);
    const hours = Number(daysMatch[2]);
    const minutes = Number(daysMatch[3]);
    const seconds = Number(daysMatch[4]);
    return (((days * 24 + hours) * 60 + minutes) * 60) + seconds;
  }

  const hmsMatch = trimmed.match(/^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$/);
  if (hmsMatch) {
    const hours = Number(hmsMatch[1]);
    const minutes = Number(hmsMatch[2]);
    const seconds = Number(hmsMatch[3]);
    return ((hours * 60 + minutes) * 60) + seconds;
  }

  const msMatch = trimmed.match(/^(\d+):(\d{2}(?:\.\d+)?)$/);
  if (msMatch) {
    const minutes = Number(msMatch[1]);
    const seconds = Number(msMatch[2]);
    return minutes * 60 + seconds;
  }

  return null;
}

function parseSortNumber(text) {
  if (!text) return null;
  const cleaned = text.replace(/,/g, '').replace(/%$/, '');
  if (!cleaned) return null;
  const value = Number(cleaned);
  return Number.isNaN(value) ? null : value;
}

function getSortValue(cell) {
  if (!cell) return { type: 'empty', value: null };
  const raw = (cell.getAttribute('data-sort') || cell.getAttribute('data-value') || cell.textContent || '').trim();
  if (!raw || raw === '-') return { type: 'empty', value: null };

  const dateValue = parseSortDate(raw);
  if (dateValue !== null) return { type: 'number', value: dateValue };

  const timeValue = parseSortTime(raw);
  if (timeValue !== null) return { type: 'number', value: timeValue };

  const numberValue = parseSortNumber(raw);
  if (numberValue !== null) return { type: 'number', value: numberValue };

  return { type: 'text', value: raw.toLowerCase() };
}

function compareSortValues(a, b, direction) {
  const aEmpty = a.type === 'empty';
  const bEmpty = b.type === 'empty';
  if (aEmpty && bEmpty) return 0;
  if (aEmpty) return 1;
  if (bEmpty) return -1;

  if (a.type === 'number' && b.type === 'number') {
    return direction === 'asc' ? a.value - b.value : b.value - a.value;
  }
  if (a.type === 'number' && b.type !== 'number') return direction === 'asc' ? -1 : 1;
  if (a.type !== 'number' && b.type === 'number') return direction === 'asc' ? 1 : -1;

  const aText = a.value || '';
  const bText = b.value || '';
  return direction === 'asc' ? aText.localeCompare(bText) : bText.localeCompare(aText);
}

function buildRowGroups(tbody) {
  const groups = [];
  const emptyRows = [];
  const detailRows = new Map();

  tbody.querySelectorAll('tr.match-details').forEach((row) => {
    if (row.id) {
      detailRows.set(row.id, row);
    }
  });

  tbody.querySelectorAll('tr').forEach((row) => {
    if (row.classList.contains('match-details')) return;
    if (row.querySelector('.empty')) {
      emptyRows.push(row);
      return;
    }
    const groupRows = [row];
    const matchId = row.dataset.matchId;
    if (matchId) {
      const detail = detailRows.get(`details-${matchId}`);
      if (detail) {
        groupRows.push(detail);
      }
    }
    groups.push({ row, rows: groupRows });
  });

  return { groups, emptyRows };
}

function initTimelineSelector() {
  const timelineButtons = document.querySelectorAll('.timeline-btn');
  if (!timelineButtons.length) {
    return;
  }

  const timelineTbodys = document.querySelectorAll('.timeline-tbody');

  timelineButtons.forEach(button => {
    button.addEventListener('click', () => {
      const timeline = button.dataset.timeline;

      timelineButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');

      timelineTbodys.forEach(tbody => {
        if (tbody.id === `timeline-${timeline}`) {
          tbody.style.display = '';
        } else {
          tbody.style.display = 'none';
        }
      });
    });
  });
}

// Table sorting (click header to sort) + attach table nav
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.data-table th').forEach((header) => {
    header.style.cursor = 'pointer';
    header.addEventListener('click', () => {
      const table = header.closest('table');
      const tbody = table?.querySelector('tbody');
      if (!table || !tbody) return;

      const columnIndex = header.cellIndex;
      const currentDir = header.classList.contains('sort-asc')
        ? 'asc'
        : header.classList.contains('sort-desc')
          ? 'desc'
          : null;
      const nextDir = currentDir === 'asc' ? 'desc' : 'asc';

      table.querySelectorAll('th').forEach((th) => {
        th.classList.remove('sort-asc', 'sort-desc');
      });
      header.classList.add(nextDir === 'asc' ? 'sort-asc' : 'sort-desc');

      const { groups, emptyRows } = buildRowGroups(tbody);
      groups.sort((a, b) => {
        const aValue = getSortValue(a.row.cells[columnIndex]);
        const bValue = getSortValue(b.row.cells[columnIndex]);
        return compareSortValues(aValue, bValue, nextDir);
      });

      tbody.innerHTML = '';
      groups.forEach((group) => {
        group.rows.forEach((row) => tbody.appendChild(row));
      });
      emptyRows.forEach((row) => tbody.appendChild(row));
    });
  });

  attachTableNav();
  initSessionFilters();
  initTableFilters();
  initMobileNavGroups();
  initStickyFilters();
  initFilterChips();
  initTableDensity();
  initPlayerHoverCard();
  initCsrOnlineStatus();
  initTimelineSelector();
});

// Auto-refresh page every 60 seconds (configurable)
(function() {
  const refreshInterval = 60000; // 60 seconds
  let lastActivity = Date.now();
  
  // Track user activity
  ['mousemove', 'keydown', 'scroll', 'click'].forEach(event => {
    document.addEventListener(event, () => {
      lastActivity = Date.now();
    });
  });
  
  // Only refresh if user hasn't been active for 30 seconds
  setInterval(() => {
    if (Date.now() - lastActivity > 30000) {
      // Check if we're on a page that should auto-refresh
      const path = window.location.pathname;
      if (path === '/' || path === '/leaderboard' || path === '/trends') {
        location.reload();
      }
    }
  }, refreshInterval);
})();
