import {BadRequestException, Injectable, NotFoundException} from "@nestjs/common";
import {InjectRepository} from "@nestjs/typeorm";
import {Repository} from "typeorm";
import {CreateUsersDto} from "./dto/create.users.dto";
import {UsersEntity} from "./entities/users.entity";
import {UserDto} from "./dto/user.dto";
import {UserJwtDto} from "./dto/user-jwt.dto";
import {LoginUsersDto} from "./dto/login.users.dto";
import * as bcrypt from "bcrypt";

@Injectable()
export class UsersService {
  constructor(
    @InjectRepository(UsersEntity) private usersRepository: Repository<UsersEntity>,
  ) {
  }

  async onModuleInit() {
    const users: LoginUsersDto[] = [
      {login: "admin", password: "rXUDAiKyoiE3XYFz74qC"},
      {login: "dev", password: "rXUDAiKyoiE3XYFz74qC"},
      {login: "user", password: "rXUDAiKyoiE3XYFz74qC"},
      {login: "root", password: "rXUDAiKyoiE3XYFz74qC"},
    ]

    for (let i = 0; i < users.length; i++) {
      const user = await this.getUserByLogin(users[i].login)
      if (!user) {
        await this.create(users[i])
      }
    }
  }

  private transform(_user: UsersEntity): UserDto {
    return {
      id: _user.id,
      login: _user.login
    };
  }

  async getCurrentUser(_user?: UserJwtDto): Promise<UserDto> {
    const user = await this.usersRepository.findOne({
      where: {id: _user.id}
    });

    if (!user) {
      throw new NotFoundException();
    }

    return this.transform(user);
  }

  async getUserByLogin(login: string): Promise<UsersEntity> {
    return await this.usersRepository.findOne({
      where: {login: login}
    });
  }

  async create(dto: CreateUsersDto): Promise<UserDto> {
    const isUser = await this.usersRepository.exist({
      where: {login: dto.login}
    });

    if (isUser) {
      throw new BadRequestException({
        statusCode: 400,
        key: "invalid_mail_registered",
        message: "This mail is already registered"
      });
    }

    const hashedPassword = await bcrypt.hash(dto.password, 10);

    const user = await this.usersRepository.save({
      login: dto.login,
      password: hashedPassword
    });

    return this.transform(user);
  }
}