func.func @meet(%lhs: !transfer.abs_value<[!transfer.integer, !transfer.integer]>, %rhs: !transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]> {
  %lhs_lb = "transfer.get"(%lhs) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %lhs_ub = "transfer.get"(%lhs) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %rhs_lb = "transfer.get"(%rhs) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %rhs_up = "transfer.get"(%rhs) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %res_lb = "transfer.umin"(%lhs_lb, %rhs_lb) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %res_up = "transfer.umax"(%lhs_ub, %rhs_up) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %res = "transfer.make"(%res_lb, %res_ub) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>
  return %res : !transfer.abs_value<[!transfer.integer, !transfer.integer]>
}
