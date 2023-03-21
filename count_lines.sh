#!/bin/bash
py_lines=$(git ls-files | grep -E ".py" | xargs cat | wc -l)
cpp_lines=$(git ls-files | grep -E ".cpp" | xargs cat | wc -l)
hpp_lines=$(git ls-files | grep -E ".hpp" | xargs cat | wc -l)
kt_lines=$(git ls-files | grep -E ".kt" | xargs cat | wc -l)
sh_lines=$(git ls-files | grep -E ".sh" | xargs cat | wc -l)
total_lines=$((py_lines+cpp_lines+hpp_lines+kt_lines+sh_lines))
echo "This is the lines-of-code summary:"
echo "----------------------------------"
echo "Python .py lines:  $py_lines"
echo "C++ .cpp lines:    $cpp_lines"
echo "Header .hpp lines: $hpp_lines"
echo "Kotlin .kt lines:  $kt_lines"
echo "Bash .sh lines:    $sh_lines"
echo "Total lines:       $total_lines"
echo "----------------------------------"