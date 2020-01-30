---


---

<h1 id="grid-csr">Grid-CSR</h1>
<ul>
<li>Grid : Blocked Matrix</li>
<li>Grid-CSR : Grid-divided CSR structure</li>
</ul>
<h2 id="csr-types-for-grid-csr">CSR Types for Grid-CSR</h2>
<h3 id="csr-type-1">CSR Type 1</h3>
<pre><code>ptr: 0 0 1 1 2 4 4 7
col: 0 1 1 2 0 2 4
</code></pre>
<h3 id="csr-type-2">CSR Type 2</h3>
<pre><code>row: 1 3 4 6
ptr: 0 0 1 1 2 4 4 7
col: 0 1 1 2 0 2 4
</code></pre>
<h3 id="csr-type-3">CSR Type 3</h3>
<pre><code>row: 1 3 4 6
ptr: 0 1 2 4 7
col: 0 1 1 2 0 2 4
</code></pre>

