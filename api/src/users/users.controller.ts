import {
  Controller,
  Get,
  Request, UseGuards
} from "@nestjs/common";
import { ApiOperation, ApiTags } from "@nestjs/swagger";
import { UsersService } from "./users.service";
import { JwtAuthGuard } from "../auth/guards/jwt-auth.guard";

@ApiTags("Профиль")
@Controller("profile")
export class UsersController {
  constructor(
    private readonly usersService: UsersService
  ) {
  }

  @ApiOperation({ summary: "Получение информации пользователя (Only auth)" })
  @UseGuards(JwtAuthGuard)
  @Get()
  getCurrentUser(@Request() req: Request) {
    return this.usersService.getCurrentUser(req["user"]);
  }
}