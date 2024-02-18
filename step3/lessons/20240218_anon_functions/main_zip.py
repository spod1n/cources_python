student_names = ["StudentName1", "StudentName2", "StudentName3", "StudentName4"]
grades = [85, 90, 78, 92]

students_and_grades = dict(zip(student_names, grades))
average_grade = sum(grades) / len(grades)

print("Dictionary of student names and grades:", students_and_grades)
print("Average grade of students:", average_grade)
