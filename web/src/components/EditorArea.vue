<script setup>
// The centre pane's placeholder (ed-02): an empty tab strip, the read-only
// badge and the empty state. ed-03 fills the body with the selected file and
// ed-04 replaces it with CodeMirror; the header chrome below is what those
// steps keep.
//
// The badge is deliberate, not decoration: a tab labelled "Editor" promises an
// editable surface, and nothing in this lot writes to disk — there is no write
// endpoint at all. It disappears when the write layer lands.
</script>

<template>
  <section class="editor-area" aria-label="File editor">
    <header class="editor-area-head">
      <!-- ed-04 fills this strip with one tab per open file and puts the
           `role="tablist"` on it then: an empty tablist would be announced as
           a widget with no choices. -->
      <div class="editor-tabs" aria-label="Open files"></div>
      <span class="editor-badge badge gray" title="Files are opened read-only in this version">
        read-only
      </span>
    </header>

    <div class="editor-body">
      <div class="stream-empty">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <path d="M14 2v6h6" />
        </svg>
        <p class="stream-empty-title">Open a file</p>
        <p class="caption">
          Pick one in the tree on the left — its contents appear here.
        </p>
      </div>
    </div>
  </section>
</template>
