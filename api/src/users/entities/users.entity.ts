import {
  Column,
  CreateDateColumn,
  Entity, Index,
  PrimaryGeneratedColumn, UpdateDateColumn
} from "typeorm";

@Entity("users")
export class UsersEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Index()
  @Column({ unique: true })
  login: string;

  @Column()
  password: string;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}